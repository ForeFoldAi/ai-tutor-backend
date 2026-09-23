"""Public chapter-aware QA entrypoints (sync-style async + stream)."""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, AsyncIterator

from app.core.student_messages import ANSWER_NOT_IN_CHAPTER
from app.config import (
    CONTEXT_CHAR_BUDGET,
    ENABLE_LLM_IMAGE_SELECT,
    IMAGE_RETRIEVAL_TIMEOUT_SEC,
    RETRIEVAL_K,
    TOP_RELATED_IMAGES,
)
from app.services.chat_service.answer_types import (
    _apply_agent_mode,
    _is_mathematics_subject,
    _is_science_subject,
    _normalize_agent_mode,
    _resolve_answer_type,
    _structure_tier,
    _voice_should_use_text_format,
    detect_answer_type,
)
from app.services.vector_service import InMemoryDocVectorStore
from app.services.chat_service.dialogue import (
    _is_affirmation_followup,
    _is_personal_dialogue_response,
)
from app.services.chat_service.images import (
    _image_top_n_for_scope,
    _log_image_stage,
    _select_related_images_for_answer,
    figure_hint_block_from_docs,
)
from app.services.chat_service.llm import _ensure_mistral_config

# Patch-compat: tests patch app.services.chat_service._stream_mistral_async / _fetch_related_images
# / _call_mistral_async — look up on the package at call time (not a local import binding).
def _cs():
    from app.services import chat_service as m
    return m
from app.services.chat_service.postprocess import (
    _DIRECT_ANSWER_TOKEN_LIMIT,
    _MAIN_SECTION_TOKEN_LIMIT,
    _expand_short_answer,
    _finalize_direct_answer,
    _join_context_within_budget,
    normalize_direct_answer_prose,
    strip_embedded_figure_lines,
)
from app.services.chat_service.prompts import (
    PROMPT,
    _build_chat_messages,
    _build_prompt,
    _mentor_profile_after_turn,
    _mentor_profile_for_turn,
    _prepare_math_engine_block,
    _safe_build_chat_messages,
    build_session_greeting,
)
from app.services.chat_service.postprocess import _mistral_token_limit_for_answer_type

logger = logging.getLogger(__name__)


def _section_instruction_with_figures(
    section_instruction: str,
    docs: list,
    *,
    skip_retrieval: bool,
) -> str:
    if skip_retrieval:
        return ""
    hint = figure_hint_block_from_docs(docs)
    base = (section_instruction or "").strip()
    if not hint:
        return base
    return f"{base}\n\n{hint}".strip() if base else hint


def _best_chunk_fallback(query: str, docs: list) -> str:
    from app.services.query_match import document_page, keyword_match_score

    best_text, best_score, best_page = "", -1, 10**9
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        score = keyword_match_score(query, text)
        pg = document_page(d)
        if score > best_score or (score == best_score and pg < best_page):
            best_score, best_page, best_text = score, pg, text

    if not best_text:
        return ANSWER_NOT_IN_CHAPTER

    parts = re.split(r"(?<=[.?!])\s+", best_text)
    snippet = " ".join(parts[:3]).strip()
    return snippet or best_text[:600].strip()


async def chapter_aware_qa(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None = None,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    chapter_names: list[str] | None = None,
    conversation_history: list[dict] | None = None,
    conversation_memory: Any | None = None,
    student_name: str = "",
    student_key: str = "",
    images_only: bool = False,
    agent_mode: str | None = None,
) -> tuple[str, list[dict], dict | None, dict | None]:
    """
    Retrieve relevant chunks from ChromaDB and answer via Mistral (async).

    Returns ``(answer_text, related_images, math_lesson, science_experiment)``.
    """
    from app.services.math_lesson.service import safe_finalize_math_answer
    from app.services.science_experiment.service import finalize_science_answer
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.conversation_memory import prepare_conversation_inputs
    from app.services.section_retrieval import retrieve_for_tutor_query
    from app.services.chapter_scope import resolve_chapter_awareness_turn
    from app.services.section_heading import HeadingScope
    from app.services.voice_ack import resolve_dialogue_act

    recent_hist, session_mem = prepare_conversation_inputs(
        conversation_history,
        memory=conversation_memory,
        chapter=chapter,
    )

    dialogue_act_resolved = resolve_dialogue_act(query)
    skip_retrieval = bool(dialogue_act_resolved)

    conv = resolve_conversation_context(
        query,
        conversation_history=recent_hist,
        chapter=chapter,
        memory=session_mem,
    )
    _apply_agent_mode(conv, agent_mode)
    retrieval_query = conv.retrieval_query or query

    if skip_retrieval:
        docs, scope, section_instruction = [], HeadingScope(kind="general"), ""
    else:
        docs, scope, section_instruction = retrieve_for_tutor_query(
            retrieval_query,
            collection_name=collection_name,
            chapter_ids=chapter_ids,
            chapter_names=chapter_names,
        )

    # Topics-left questions: answer from coverage store (no LLM needed).
    if student_key and student_key.isdigit() and chapter_ids and not skip_retrieval:
        try:
            from app.core.database import SessionLocal
            from app.modules.student_learning.topic_progress import try_topics_left_reply

            db = SessionLocal()
            try:
                left_reply = try_topics_left_reply(
                    db,
                    user_id=int(student_key),
                    query=query,
                    chapter_ids=chapter_ids,
                    subject_name=subject_name,
                    chapter=chapter,
                    board=board,
                    class_level=class_level,
                )
            finally:
                db.close()
            if left_reply:
                return left_reply, [], None, None
        except Exception:
            logger.exception("topics-left reply failed")

    if skip_retrieval:
        early, effective_query, coverage_guidance = None, query, ""
    else:
        early, effective_query, _assessment, coverage_guidance = await resolve_chapter_awareness_turn(
            query,
            docs=docs,
            conversation_history=recent_hist,
            collection_name=collection_name,
            chapter_ids=chapter_ids,
            chapter_names=chapter_names,
            board=board,
            class_level=class_level,
            subject_name=subject_name,
            scope_query=retrieval_query,
        )
    if early:
        return early, [], None, None

    img_allowed = should_retrieve_images(
        conv,
        chapter_ids=chapter_ids,
        heading_scope_kind=scope.kind,
        subject_name=subject_name,
    )
    if skip_retrieval:
        img_allowed = False
    if images_only:
        related: list[dict] = []
        if chapter_ids and img_allowed and docs:
            img_top_n = _image_top_n_for_scope(scope, docs=docs)
            try:
                related = await asyncio.wait_for(
                    asyncio.to_thread(
                        _cs()._fetch_related_images,
                        scope,
                        collection_name=collection_name,
                        chapter_ids=chapter_ids,
                        chapter_names=chapter_names or [],
                        chapter=chapter,
                        query=effective_query,
                        docs=docs,
                        conversation_history=recent_hist,
                        top_n=img_top_n,
                    ),
                    timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
                )
            except Exception as exc:
                logger.warning("Image-only retrieval failed: %s", exc)
        return "", related, None, None
    if not docs and not coverage_guidance and not skip_retrieval:
        return ANSWER_NOT_IN_CHAPTER, [], None, None

    learner_snapshot, understanding_scores = await _mentor_profile_for_turn(
        student_key,
        effective_query,
        recent_hist,
        conv.resolved_topic or effective_query,
    )

    if student_key and student_key.isdigit():
        from app.services.learning_intelligence.clients.lia_client import emit_chat_user_question

        last_assistant = ""
        for turn in reversed(recent_hist or []):
            if (turn.get("role") or "").lower() == "assistant":
                last_assistant = (turn.get("content") or "").strip()
                break
        emit_chat_user_question(
            student_user_id=int(student_key),
            school_id=None,
            query=effective_query,
            subject_name=subject_name,
            chapter=chapter,
            chapter_ids=chapter_ids,
            class_level=class_level,
            board=board,
            understanding_scores=understanding_scores,
            agent_mode=agent_mode,
            last_assistant=last_assistant,
        )
        try:
            from app.modules.student_learning.topic_progress import record_turn_topic_progress

            scope_title = None
            if getattr(scope, "matched", None) is not None:
                scope_title = getattr(scope.matched, "title", None)
            record_turn_topic_progress(
                student_user_id=int(student_key),
                query=effective_query,
                chapter_ids=chapter_ids,
                subject_name=subject_name,
                chapter=chapter,
                board=board,
                class_level=class_level,
                scope_title=scope_title,
            )
        except Exception:
            logger.exception("topic progress record failed")

    context = "" if skip_retrieval else _join_context_within_budget(docs)
    img_top_n = _image_top_n_for_scope(scope, docs=docs)
    text_coverage = coverage_guidance
    if dialogue_act_resolved == "intro":
        text_coverage = (
            (coverage_guidance + "\n\n" if coverage_guidance else "")
            + "DIALOGUE: The student is introducing themselves — greet briefly and invite a "
            "lesson question. Do not say the topic is uncovered or show an a/b chapter menu."
        )
    elif dialogue_act_resolved in ("closing", "ack"):
        text_coverage = (
            (coverage_guidance + "\n\n" if coverage_guidance else "")
            + "DIALOGUE: Acknowledge warmly from conversation history only — do not re-teach "
            "or dump chapter content. Offer one next step (example, quiz, next part, wrap up)."
        )
    messages = _safe_build_chat_messages(
        effective_query,
        context,
        class_level=class_level,
        board=board,
        subject_name=subject_name,
        chapter=chapter,
        student_name=student_name,
        section_instruction=_section_instruction_with_figures(
            section_instruction, docs, skip_retrieval=skip_retrieval
        ),
        heading_scope=scope,
        conversation_history=recent_hist,
        conversation_memory=session_mem,
        chapter_coverage_guidance=text_coverage,
        learner_snapshot=learner_snapshot,
        understanding_scores=understanding_scores,
        resolved_topic=conv.resolved_topic or effective_query,
        agent_mode=agent_mode,
        student_key=student_key,
        chapter_ids=chapter_ids,
        dialogue_act=dialogue_act_resolved,
    )
    if messages is None:
        answer = _best_chunk_fallback(effective_query, docs)
        related: list[dict] = []
        if chapter_ids and img_allowed:
            try:
                related = await asyncio.wait_for(
                    asyncio.to_thread(
                        _cs()._fetch_related_images,
                        scope,
                        collection_name=collection_name,
                        chapter_ids=chapter_ids,
                        chapter_names=chapter_names or [],
                        chapter=chapter,
                        query=effective_query,
                        docs=docs,
                        conversation_history=recent_hist,
                        top_n=img_top_n,
                    ),
                    timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
                )
            except Exception as exc:
                logger.warning("Image retrieval failed (build fallback): %s", exc)
        clean, math_lesson = safe_finalize_math_answer(
            answer,
            effective_query,
            class_level=class_level,
            subject_name=subject_name,
            conversation_history=recent_hist,
        )
        return clean, related, math_lesson, None

    answer_type = _resolve_answer_type(
        effective_query,
        subject_name=subject_name,
        conversation_history=recent_hist,
        chapter=chapter,
        conv=conv,
        dialogue_act=dialogue_act_resolved,
    )
    if _normalize_agent_mode(agent_mode) == "practice":
        answer_type = "quiz"
    elif _normalize_agent_mode(agent_mode) in ("ask", "explain") and answer_type in ("quiz", "mcq"):
        answer_type = "explanation"
    token_limit = _mistral_token_limit_for_answer_type(
        answer_type,
        heading_scope=scope,
        subject_name=subject_name,
        query=effective_query,
    )

    llm_ok = False
    try:
        answer = await _cs()._call_mistral_async(messages, max_tokens=token_limit)
        answer = await _finalize_direct_answer(
            messages,
            answer,
            effective_query,
            answer_type=answer_type,
            heading_scope=scope,
            conversation_history=recent_hist,
            subject_name=subject_name,
        )
        llm_ok = True
    except FileNotFoundError:
        answer = _best_chunk_fallback(effective_query, docs)
    except Exception as exc:
        logger.error("Mistral call failed: %s: %s", type(exc).__name__, exc)
        answer = _best_chunk_fallback(effective_query, docs)

    from app.services.chapter_scope import scrub_wrong_premise_echo

    answer = scrub_wrong_premise_echo(
        answer,
        effective_query,
        docs=docs,
        chapter_names=chapter_names,
    )

    related: list[dict] = []
    if chapter_ids and img_allowed:
        try:
            related = await asyncio.wait_for(
                asyncio.to_thread(
                    _cs()._fetch_related_images,
                    scope,
                    collection_name=collection_name,
                    chapter_ids=chapter_ids,
                    chapter_names=chapter_names or [],
                    chapter=chapter,
                    query=effective_query,
                    docs=docs,
                    conversation_history=recent_hist,
                    top_n=img_top_n,
                ),
                timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
            )
            # Don't burn quota on image_select when the answer LLM already failed,
            # unless the student asked for Fig N / a chapter figure list (DB path).
            from app.services.image_service.image_intent_extractor import (
                figure_numbers_cited_in_text,
                is_chapter_figure_list_ask,
            )

            cite_or_list = (
                is_chapter_figure_list_ask(effective_query)
                or bool(figure_numbers_cited_in_text(effective_query))
                or bool(figure_numbers_cited_in_text(answer))
            )
            if llm_ok or cite_or_list:
                related = await _select_related_images_for_answer(
                    effective_query,
                    answer,
                    related,
                    retrieved_docs=docs,
                    chapter_ids=chapter_ids,
                )
            elif ENABLE_LLM_IMAGE_SELECT:
                related = []
        except asyncio.TimeoutError:
            logger.warning(
                "Image retrieval timed out after %.0fs for query=%r",
                IMAGE_RETRIEVAL_TIMEOUT_SEC,
                effective_query[:80],
            )
        except Exception as exc:
            logger.warning("Image retrieval failed: %s", exc)

    from app.services.chat_service.images import figure_inventory_text
    from app.services.image_service.image_intent_extractor import is_chapter_figure_list_ask

    if is_chapter_figure_list_ask(effective_query) and related:
        answer = figure_inventory_text(related)

    answer, math_lesson = safe_finalize_math_answer(
        answer,
        effective_query,
        class_level=class_level,
        subject_name=subject_name,
        conversation_history=recent_hist,
    )
    answer, science_experiment = finalize_science_answer(
        answer,
        effective_query,
        class_level=class_level,
        subject_name=subject_name,
    )

    await _mentor_profile_after_turn(
        student_key,
        conv.resolved_topic or effective_query,
        understanding_scores,
    )
    if student_key and student_key.isdigit():
        from app.services.learning_intelligence.clients.lia_client import emit_chat_assistant_response

        emit_chat_assistant_response(
            student_user_id=int(student_key),
            subject_name=subject_name,
            chapter=chapter,
            topic=conv.resolved_topic or effective_query,
            agent_mode=agent_mode,
        )
    return answer, related, math_lesson, science_experiment


async def chapter_aware_qa_stream(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None = None,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    chapter_names: list[str] | None = None,
    emit_related_images: Callable[[list[dict]], Awaitable[None]] | None = None,
    emit_clean_answer: Callable[[str], Awaitable[None]] | None = None,
    emit_math_lesson: Callable[[dict | None, str], Awaitable[None]] | None = None,
    emit_science_experiment: Callable[[dict | None, str], Awaitable[None]] | None = None,
    conversation_history: list[dict] | None = None,
    conversation_memory: Any | None = None,
    student_name: str = "",
    student_key: str = "",
    voice_mode: bool = False,
    tutor_state: str = "TEACHING",
    understanding_scores: dict | None = None,
    learner_snapshot: dict | None = None,
    pipeline_timing: Any | None = None,
    agent_mode: str | None = None,
    quiz_pending: bool = False,
    quiz_question: str = "",
    quiz_attempts: int = 0,
    explained_points: list[str] | None = None,
    nest_intent: str | None = None,
    dialogue_act: str | None = None,
    filler_phrase_played: str | None = None,
    affect_trajectory: list[str] | None = None,
    voice_metadata_out: dict | None = None,
) -> AsyncIterator[str]:
    """
    Streaming version of chapter_aware_qa.

    Optionally invokes *emit_related_images* when ranked images are ready
    (usually during the first tokens, without blocking retrieval).
    """
    from app.config import CONTEXT_CHAR_BUDGET, EARLY_IMAGE_MIN_CHARS, RETRIEVAL_K, VOICE_EARLY_IMAGE_MIN_CHARS, VOICE_MAX_TOKENS, VOICE_CONTEXT_CHAR_BUDGET, VOICE_RETRIEVAL_K
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.conversation_memory import format_memory_for_prompt, prepare_conversation_inputs
    from app.services.image_service.textbook_image_retrieval import early_related_images_for_query
    from app.services.math_lesson.service import safe_finalize_math_answer
    from app.services.science_experiment.service import finalize_science_answer
    from app.services.section_retrieval import retrieve_for_tutor_query, voice_context_budget

    retrieval_k = VOICE_RETRIEVAL_K if voice_mode else RETRIEVAL_K
    context_budget = VOICE_CONTEXT_CHAR_BUDGET if voice_mode else CONTEXT_CHAR_BUDGET

    if voice_mode:
        from app.services.voice_stt_postprocess import (
            INCOMPLETE_UTTERANCE_REPLY,
            is_incomplete_voice_utterance,
        )

        if is_incomplete_voice_utterance(query):
            if emit_related_images:
                await emit_related_images([])
            yield INCOMPLETE_UTTERANCE_REPLY
            return

    from app.services.voice_ack import resolve_dialogue_act

    # LLM-first: closing/ack/intro skip chapter RAG (voice + text) but still call the model.
    dialogue_act_resolved = resolve_dialogue_act(
        query, dialogue_act=dialogue_act, quiz_pending=quiz_pending
    )
    skip_retrieval = bool(dialogue_act_resolved)

    from app.services.chapter_scope import resolve_chapter_awareness_turn

    recent_hist, session_mem = prepare_conversation_inputs(
        conversation_history,
        memory=conversation_memory,
        chapter=chapter,
    )

    voice_ctx: dict[str, Any] = {}
    student_affect = None
    if voice_mode:
        last_assistant_pre = ""
        for turn in reversed(recent_hist or []):
            if (turn.get("role") or "").lower() == "assistant":
                last_assistant_pre = (turn.get("content") or "").strip()
                break
        from app.services.student_affect import (
            evaluate_student_affect_async,
            merge_affect_trajectory,
            rapport_from_trajectory,
        )

        student_affect = await evaluate_student_affect_async(
            query,
            last_assistant=last_assistant_pre,
            tutor_state=tutor_state or "TEACHING",
        )
        understanding_scores = student_affect.to_understanding_scores()
        merged_trajectory = merge_affect_trajectory(
            affect_trajectory, student_affect.primary
        )
        session_mem.affect_trajectory = merged_trajectory
        if student_affect.primary in ("excited", "affirmation", "curious"):
            session_mem.last_positive_moment = query[:120]
        session_mem.engagement_trend = (
            session_mem.engagement_trend * 0.7 + student_affect.engagement * 0.3
        )
        voice_ctx["affect"] = student_affect
        voice_ctx["affect_trajectory"] = merged_trajectory
        voice_ctx["rapport_hint"] = rapport_from_trajectory(merged_trajectory)
        if student_key and learner_snapshot is None:
            from app.services.learner_profile import load_learner_profile

            profile = await load_learner_profile(student_key)
            learner_snapshot = profile.snapshot().__dict__

    conv = resolve_conversation_context(
        query,
        conversation_history=recent_hist,
        chapter=chapter,
        memory=session_mem,
    )
    _apply_agent_mode(conv, agent_mode)
    retrieval_query = conv.retrieval_query or query
    if voice_mode:
        context_budget = voice_context_budget(context_budget, retrieval_query)

    from app.services.section_heading import HeadingScope

    if skip_retrieval:
        docs, scope, section_instruction = [], HeadingScope(kind="general"), ""
        if pipeline_timing is not None:
            pipeline_timing.mark_rag_done()
    elif voice_mode:
        docs, scope, section_instruction = await asyncio.to_thread(
            retrieve_for_tutor_query,
            retrieval_query,
            collection_name=collection_name,
            chapter_ids=chapter_ids,
            chapter_names=chapter_names,
            k=retrieval_k,
        )
        if pipeline_timing is not None:
            pipeline_timing.mark_rag_done()
    else:
        docs, scope, section_instruction = retrieve_for_tutor_query(
            retrieval_query,
            collection_name=collection_name,
            chapter_ids=chapter_ids,
            chapter_names=chapter_names,
            k=retrieval_k,
        )

    if student_key and student_key.isdigit() and chapter_ids and not skip_retrieval:
        try:
            from app.core.database import SessionLocal
            from app.modules.student_learning.topic_progress import try_topics_left_reply

            db = SessionLocal()
            try:
                left_reply = try_topics_left_reply(
                    db,
                    user_id=int(student_key),
                    query=query,
                    chapter_ids=chapter_ids,
                    subject_name=subject_name,
                    chapter=chapter,
                    board=board,
                    class_level=class_level,
                )
            finally:
                db.close()
            if left_reply:
                if emit_related_images:
                    await emit_related_images([])
                yield left_reply
                return
        except Exception:
            logger.exception("topics-left reply failed (stream)")

    if skip_retrieval:
        early, effective_query, coverage_guidance = None, query, ""
    else:
        early, effective_query, _assessment, coverage_guidance = await resolve_chapter_awareness_turn(
            query,
            docs=docs,
            conversation_history=recent_hist,
            collection_name=collection_name,
            chapter_ids=chapter_ids,
            chapter_names=chapter_names,
            board=board,
            class_level=class_level,
            subject_name=subject_name,
            scope_query=retrieval_query,
            spoken=voice_mode,
        )
    if early:
        if emit_related_images:
            await emit_related_images([])
        yield early
        return

    use_text_format = _voice_should_use_text_format(
        subject_name,
        voice_mode=voice_mode,
        understanding_scores=understanding_scores,
        query=effective_query,
        heading_scope=scope,
    )
    if skip_retrieval:
        use_text_format = False
    voice_live_teaching = voice_mode and not use_text_format

    img_allowed = should_retrieve_images(
        conv,
        chapter_ids=chapter_ids,
        heading_scope_kind=scope.kind,
        subject_name=subject_name,
        voice_mode=voice_mode,
    )
    if skip_retrieval:
        img_allowed = False
        if emit_related_images:
            await emit_related_images([])
    img_top_n = _image_top_n_for_scope(scope, docs=docs)
    if not docs and not coverage_guidance and not skip_retrieval:
        if emit_related_images:
            await emit_related_images([])
        yield ANSWER_NOT_IN_CHAPTER
        return

    if emit_related_images and not img_allowed:
        await emit_related_images([])

    if student_key and student_key.isdigit():
        from app.services.learning_intelligence.clients.lia_client import emit_chat_user_question

        last_assistant = ""
        for turn in reversed(recent_hist or []):
            if (turn.get("role") or "").lower() == "assistant":
                last_assistant = (turn.get("content") or "").strip()
                break
        emit_chat_user_question(
            student_user_id=int(student_key),
            school_id=None,
            query=effective_query,
            subject_name=subject_name,
            chapter=chapter,
            chapter_ids=chapter_ids,
            class_level=class_level,
            board=board,
            understanding_scores=understanding_scores,
            agent_mode=agent_mode,
            last_assistant=last_assistant,
        )
        try:
            from app.modules.student_learning.topic_progress import record_turn_topic_progress

            scope_title = None
            if getattr(scope, "matched", None) is not None:
                scope_title = getattr(scope.matched, "title", None)
            record_turn_topic_progress(
                student_user_id=int(student_key),
                query=effective_query,
                chapter_ids=chapter_ids,
                subject_name=subject_name,
                chapter=chapter,
                board=board,
                class_level=class_level,
                scope_title=scope_title,
            )
        except Exception:
            logger.exception("topic progress record failed (stream)")

    context = _join_context_within_budget(docs, char_budget=context_budget)
    if skip_retrieval:
        context = ""
    logger.info(
        "[VOICE-TURN] mode=%s docs=%d ctx_chars=%d sufficient=%s q=%r rq=%r act=%r",
        "voice" if voice_live_teaching else ("voice-text" if voice_mode else "chat"),
        len(docs or []),
        len(context),
        bool(docs),
        (query or "")[:80],
        (retrieval_query or "")[:80],
        dialogue_act_resolved,
    )

    if not voice_live_teaching and student_key and learner_snapshot is None:
        learner_snapshot, understanding_scores = await _mentor_profile_for_turn(
            student_key,
            effective_query,
            recent_hist,
            conv.resolved_topic or effective_query,
        )

    if voice_live_teaching:
        from app.services.voice_tutor import (
            LearnerProfileSnapshot,
            ReplyIntent,
            TutorState,
            UnderstandingScores,
            build_voice_mistral_messages,
            classify_reply_intent,
            next_tutor_state,
            topic_key,
            update_quiz_state,
        )

        try:
            tutor_st = TutorState(tutor_state)
        except ValueError:
            tutor_st = TutorState.TEACHING
        scores = understanding_scores or {}
        understanding = UnderstandingScores(
            understanding=float(scores.get("understanding", 0.5)),
            confidence=float(scores.get("confidence", 0.5)),
            confusion=float(scores.get("confusion", 0.0)),
            is_affirmation=bool(scores.get("is_affirmation")),
            wants_expansion=bool(scores.get("wants_expansion")),
            wants_quiz=bool(scores.get("wants_quiz")),
        )
        reply_intent = classify_reply_intent(query, quiz_pending=quiz_pending)
        if dialogue_act_resolved == "closing":
            reply_intent = ReplyIntent.CLOSING
        elif dialogue_act_resolved == "ack":
            # Soft react — reuse CLOSING guidance shape via dialogue_act in prompts.
            reply_intent = ReplyIntent.CLOSING if not quiz_pending else ReplyIntent.UNCLEAR
        voice_ctx["tutor_st"] = tutor_st
        voice_ctx["understanding"] = understanding
        voice_ctx["reply_intent"] = reply_intent
        voice_ctx["explained_points"] = list(explained_points or [])
        voice_ctx["dialogue_act"] = dialogue_act_resolved
        learner = (
            LearnerProfileSnapshot(**learner_snapshot)
            if learner_snapshot
            else None
        )
        messages = build_voice_mistral_messages(
            # Always the student's raw utterance — retrieval may use a different
            # string via retrieval_query / scope, but never silently swap this.
            query,
            context if not skip_retrieval else "",
            class_level=class_level,
            board=board,
            subject_name=subject_name,
            chapter=chapter,
            student_name=student_name,
            conversation_history=recent_hist,
            tutor_state=tutor_st,
            understanding=understanding,
            learner=learner,
            expand_deep=(
                False
                if skip_retrieval
                else (understanding.wants_expansion or understanding.confusion >= 0.55)
            ),
            quiz_pending=quiz_pending,
            quiz_question=quiz_question,
            quiz_attempts=quiz_attempts,
            explained_points=explained_points,
            reply_intent=reply_intent,
            student_affect=student_affect,
            nest_intent=nest_intent,
            dialogue_act=dialogue_act_resolved,
            filler_phrase_played=filler_phrase_played,
        )
        if coverage_guidance:
            messages[0]["content"] = messages[0]["content"] + "\n\n" + coverage_guidance
        rapport = voice_ctx.get("rapport_hint") or ""
        if rapport:
            messages[0]["content"] = messages[0]["content"] + "\n\n" + rapport
        if session_mem.turn_count or session_mem.conversation_summary:
            mem_block = format_memory_for_prompt(session_mem)
            if mem_block:
                messages[0]["content"] = messages[0]["content"] + "\n\n" + mem_block
        if student_key and student_key.isdigit():
            from app.services.learning_intelligence.clients.lia_client import get_guidance_for_turn_sync

            lia = get_guidance_for_turn_sync(
                student_user_id=int(student_key),
                query=effective_query,
                topic=conv.resolved_topic or effective_query,
                subject_name=subject_name,
                chapter=chapter,
                chapter_ids=chapter_ids,
                class_level=class_level,
                board=board,
                agent_mode=agent_mode,
                understanding_scores=understanding_scores,
            )
            if lia and lia.get("prompt_instructions"):
                messages[0]["content"] = messages[0]["content"] + "\n\n" + lia["prompt_instructions"]
    else:
        text_coverage = coverage_guidance
        if dialogue_act_resolved == "intro":
            text_coverage = (
                (coverage_guidance + "\n\n" if coverage_guidance else "")
                + "DIALOGUE: The student is introducing themselves — greet briefly and invite a "
                "lesson question. Do not say the topic is uncovered or show an a/b chapter menu."
            )
        elif dialogue_act_resolved in ("closing", "ack"):
            text_coverage = (
                (coverage_guidance + "\n\n" if coverage_guidance else "")
                + "DIALOGUE: Acknowledge warmly from conversation history only — do not re-teach "
                "or dump chapter content. Offer one next step (example, quiz, next part, wrap up)."
            )
        messages = _safe_build_chat_messages(
            effective_query,
            context,
            class_level=class_level,
            board=board,
            subject_name=subject_name,
            chapter=chapter,
            student_name=student_name,
            section_instruction=_section_instruction_with_figures(
                section_instruction, docs, skip_retrieval=skip_retrieval
            ),
            heading_scope=scope,
            conversation_history=recent_hist,
            conversation_memory=session_mem,
            chapter_coverage_guidance=text_coverage,
            learner_snapshot=learner_snapshot,
            understanding_scores=understanding_scores,
            resolved_topic=conv.resolved_topic or effective_query,
            agent_mode=agent_mode,
            student_key=student_key,
            chapter_ids=chapter_ids,
            dialogue_act=dialogue_act_resolved,
        )

    if messages is None:
        fb = _best_chunk_fallback(effective_query, docs)
        if emit_related_images:
            await emit_related_images([])
        yield fb
        return

    last_imgs: list[dict] = []
    images_emitted = False
    early_image_min = VOICE_EARLY_IMAGE_MIN_CHARS if voice_live_teaching else EARLY_IMAGE_MIN_CHARS
    answer_type = _resolve_answer_type(
        effective_query,
        subject_name=subject_name,
        conversation_history=recent_hist,
        chapter=chapter,
        conv=conv,
        dialogue_act=dialogue_act_resolved,
    )
    if _normalize_agent_mode(agent_mode) == "practice":
        answer_type = "quiz"
    elif _normalize_agent_mode(agent_mode) in ("ask", "explain") and answer_type in ("quiz", "mcq"):
        answer_type = "explanation"
    voice_token_limit = VOICE_MAX_TOKENS if voice_live_teaching else None
    if voice_token_limit is None:
        voice_token_limit = _mistral_token_limit_for_answer_type(
            answer_type,
            heading_scope=scope,
            subject_name=subject_name,
            query=effective_query,
        )

    async def _retrieve_images_for_answer(
        answer_text: str, *, llm_ok: bool = True
    ) -> list[dict]:
        if not chapter_ids or not img_allowed:
            return []
        from app.services.image_service.image_intent_extractor import (
            figure_numbers_cited_in_text,
            is_chapter_figure_list_ask,
        )

        cite_or_list = (
            is_chapter_figure_list_ask(query)
            or bool(figure_numbers_cited_in_text(query))
            or bool(figure_numbers_cited_in_text(answer_text))
        )
        # ChatGPT-style select needs a real tutor answer; don't burn quota after LLM 429.
        # Fig N / list-ask still resolve from the chapter DB without LLM image select.
        if ENABLE_LLM_IMAGE_SELECT and not llm_ok and not cite_or_list:
            return []
        try:
            candidates = await asyncio.wait_for(
                asyncio.to_thread(
                    _cs()._fetch_related_images,
                    scope,
                    collection_name=collection_name,
                    chapter_ids=chapter_ids,
                    chapter_names=chapter_names or [],
                    chapter=chapter,
                    query=query,
                    docs=docs,
                    conversation_history=recent_hist,
                    top_n=img_top_n,
                ),
                timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
            )
            return await _select_related_images_for_answer(
                query,
                answer_text,
                candidates,
                retrieved_docs=docs,
                chapter_ids=chapter_ids,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Image retrieval timed out after %.0fs (stream end)",
                IMAGE_RETRIEVAL_TIMEOUT_SEC,
            )
            return []
        except Exception as exc:
            logger.warning("Image retrieval failed (stream end): %s", exc)
            return []

    # Bootstrap image search from retrieved textbook context while the LLM streams.
    # When LLM select is on: skip early/bootstrap emits (single final emit after answer).
    from app.services.image_service.textbook_image_retrieval import _context_excerpt_from_docs

    bootstrap_ctx = _context_excerpt_from_docs(docs)[:2000]
    img_task: asyncio.Task[list[dict]] | None = None
    early_img_task: asyncio.Task[list[dict]] | None = None
    emit_early_task: asyncio.Task[None] | None = None
    if chapter_ids and img_allowed and not ENABLE_LLM_IMAGE_SELECT:
        early_img_task = asyncio.create_task(
            asyncio.to_thread(
                early_related_images_for_query,
                chapter_ids,
                docs,
                query,
                top_n=img_top_n,
                conversation_history=recent_hist,
                chapter_single=chapter,
            )
        )
        img_task = asyncio.create_task(_retrieve_images_for_answer(bootstrap_ctx))

    async def _emit_early_images_when_ready() -> None:
        nonlocal images_emitted, last_imgs
        if early_img_task is None or not emit_related_images or not img_allowed:
            return
        try:
            imgs = await early_img_task
        except Exception:
            imgs = []
        if imgs and not images_emitted:
            last_imgs = imgs
            _log_image_stage("early_emit", imgs)
            await emit_related_images(imgs)
            images_emitted = True

    if chapter_ids and img_allowed and early_img_task is not None:
        emit_early_task = asyncio.create_task(_emit_early_images_when_ready())

    async def _cancel_bg_tasks() -> None:
        for task in (emit_early_task, img_task, early_img_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (emit_early_task, img_task, early_img_task):
            if task is None:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def _maybe_emit_bootstrap_images(answer_so_far: str = "") -> None:
        nonlocal images_emitted, last_imgs
        if images_emitted or not emit_related_images or not img_allowed:
            return
        if early_img_task is not None and early_img_task.done():
            try:
                last_imgs = early_img_task.result()
            except Exception:
                last_imgs = []
            if last_imgs:
                _log_image_stage("early_emit", last_imgs)
                await emit_related_images(last_imgs)
                images_emitted = True
                return
        min_chars = early_image_min
        if len(answer_so_far) < min_chars:
            return
        if img_task is None or not img_task.done():
            return
        try:
            last_imgs = img_task.result()
        except Exception:
            last_imgs = []
        if last_imgs:
            _log_image_stage("bootstrap_emit", last_imgs)
            await emit_related_images(last_imgs)
            images_emitted = True

    try:
        full_answer: list[str] = []
        llm_ok = False
        stream_temperature = None
        if voice_live_teaching and student_affect is not None:
            from app.config import LLM_TEMPERATURE, VOICE_LLM_TEMPERATURE_DIALOGUE

            dialogue_primaries = {
                "greeting",
                "closing",
                "personal",
                "affirmation",
                "excited",
                "bored",
                "frustrated",
            }
            if getattr(student_affect, "primary", "") in dialogue_primaries:
                stream_temperature = VOICE_LLM_TEMPERATURE_DIALOGUE
            else:
                stream_temperature = LLM_TEMPERATURE
        async for token in _cs()._stream_mistral_async(
            messages,
            max_tokens=voice_token_limit,
            feature="voice" if voice_mode else "chat",
            temperature=stream_temperature,
        ):
            full_answer.append(token)
            await _maybe_emit_bootstrap_images("".join(full_answer))
            yield token
        answer_text = "".join(full_answer)
        llm_ok = bool((answer_text or "").strip())
        # Empty LLM stream → textbook snippet so the UI never shows a blank bubble.
        if not llm_ok:
            fb = _best_chunk_fallback(effective_query, docs)
            if fb.strip():
                answer_text = fb
                yield fb
            else:
                answer_text = ANSWER_NOT_IN_CHAPTER
                yield ANSWER_NOT_IN_CHAPTER
        if not voice_live_teaching:
            original_len = len(answer_text)
            try:
                answer_text = await _finalize_direct_answer(
                    messages,
                    answer_text,
                    effective_query,
                    answer_type=answer_type,
                    heading_scope=scope,
                    conversation_history=recent_hist,
                    subject_name=subject_name,
                )
                if emit_clean_answer and answer_text and len(answer_text) != original_len:
                    await emit_clean_answer(answer_text)
                elif len(answer_text) > original_len:
                    supplement = answer_text[original_len:]
                    if supplement:
                        yield supplement
                elif len(answer_text) < original_len:
                    # ponytail: shrink replaced streamed text — client keeps full stream today;
                    # cache stores the shorter final answer.
                    pass
            except Exception as exc:
                logger.warning("Stream answer finalize failed: %s", exc)
        from app.services.chapter_scope import scrub_wrong_premise_echo

        answer_text = scrub_wrong_premise_echo(
            answer_text,
            effective_query,
            docs=docs,
            chapter_names=chapter_names,
        )
        math_lesson = None
        science_experiment = None
        should_finalize_math = use_text_format or (
            voice_mode and _is_mathematics_subject(subject_name)
        )
        should_finalize_science = use_text_format or (
            voice_mode and _is_science_subject(subject_name)
        )
        if should_finalize_math:
            from app.services.chapter_scope import detect_chapter_scope_choice

            is_scope_choice = (
                detect_chapter_scope_choice(effective_query, conversation_history) is not None
            )
            allow_fallback = not is_scope_choice and (use_text_format or voice_live_teaching)
            streamed_prose = answer_text
            clean_answer, math_lesson = safe_finalize_math_answer(
                answer_text,
                effective_query,
                class_level=class_level,
                subject_name=subject_name,
                allow_fallback=allow_fallback,
                conversation_history=recent_hist,
            )
            # Never ship an empty clean_answer — incomplete ```math-lesson fences can wipe prose.
            display_answer = (clean_answer or "").strip() or streamed_prose
            if math_lesson and emit_math_lesson:
                await emit_math_lesson(math_lesson, display_answer)
            if use_text_format or voice_live_teaching:
                answer_text = display_answer
        if should_finalize_science:
            streamed_prose = answer_text
            clean_answer, science_experiment = finalize_science_answer(
                answer_text,
                effective_query,
                class_level=class_level,
                subject_name=subject_name,
            )
            display_answer = (clean_answer or "").strip() or streamed_prose
            if science_experiment and emit_science_experiment:
                await emit_science_experiment(science_experiment, display_answer)
            if use_text_format or (voice_mode and _is_science_subject(subject_name)):
                answer_text = display_answer
        if img_task is not None and not img_task.done():
            img_task.cancel()
            try:
                await img_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        final_imgs = await _retrieve_images_for_answer(answer_text, llm_ok=llm_ok)
        last_imgs = final_imgs
        from app.services.chat_service.images import figure_inventory_text
        from app.services.image_service.image_intent_extractor import is_chapter_figure_list_ask

        if is_chapter_figure_list_ask(effective_query) and final_imgs:
            inv = figure_inventory_text(final_imgs)
            answer_text = inv
            if emit_clean_answer:
                await emit_clean_answer(inv)
        _log_image_stage("final_emit", final_imgs)
        if emit_related_images:
            await emit_related_images(final_imgs)
        if voice_live_teaching and voice_metadata_out is not None and voice_ctx:
            from app.services.voice_tutor import (
                ReplyIntent,
                next_tutor_state,
                topic_key,
            )

            aff = voice_ctx.get("affect")
            tutor_st = voice_ctx.get("tutor_st")
            understanding = voice_ctx.get("understanding")
            reply_intent = voice_ctx.get("reply_intent")
            if tutor_st and understanding:
                new_state = next_tutor_state(
                    current=tutor_st,
                    scores=understanding,
                    assistant_reply=answer_text,
                )
                explained_out = list(voice_ctx.get("explained_points") or [])
                if reply_intent == ReplyIntent.NEW_QUESTION:
                    key = topic_key(query)
                    if key and key not in explained_out:
                        explained_out.append(key)
                voice_metadata_out["tutor_state"] = new_state.value
                voice_metadata_out["explained_points"] = explained_out[-20:]
                if aff is not None:
                    voice_metadata_out["affect_summary"] = aff.summary()
                    voice_metadata_out["affect_hint"] = aff.to_student_hint()
                    voice_metadata_out["affect_primary"] = aff.primary
                if voice_ctx.get("affect_trajectory"):
                    voice_metadata_out["affect_trajectory"] = voice_ctx["affect_trajectory"]
                logger.info(
                    "voice_turn affect=%s tutor_state=%s source=%s",
                    getattr(aff, "primary", "neutral"),
                    new_state.value,
                    getattr(aff, "source", "n/a"),
                )
        if answer_text and not voice_live_teaching:
            if student_key and understanding_scores:
                await _mentor_profile_after_turn(
                    student_key,
                    conv.resolved_topic or effective_query,
                    understanding_scores,
                )
                if student_key.isdigit():
                    from app.services.learning_intelligence.clients.lia_client import (
                        emit_chat_assistant_response,
                    )

                    emit_chat_assistant_response(
                        student_user_id=int(student_key),
                        subject_name=subject_name,
                        chapter=chapter,
                        topic=conv.resolved_topic or effective_query,
                        agent_mode=agent_mode,
                    )
    except FileNotFoundError:
        fb = _best_chunk_fallback(query, docs)
        last_imgs = await _retrieve_images_for_answer(fb, llm_ok=False)
        if emit_related_images:
            await emit_related_images(last_imgs)
        yield fb
    except Exception as exc:
        logger.error("Mistral stream failed: %s: %s", type(exc).__name__, exc)
        fb = _best_chunk_fallback(query, docs)
        last_imgs = await _retrieve_images_for_answer(fb, llm_ok=False)
        if emit_related_images:
            await emit_related_images(last_imgs)
        yield fb
    finally:
        await _cancel_bg_tasks()


def get_qa_chain(vectorstore):
    """
    Backward-compatible wrapper for the old main.py ``/chat`` endpoint.
    Returns a sync-compatible object whose ``.run()`` blocks the caller
    (acceptable only for the legacy /chat route).
    """

    class _QARunnable:
        def run(self, query: str) -> str:
            import asyncio

            retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
            docs = retriever.invoke(query)

            if isinstance(vectorstore, InMemoryDocVectorStore):
                def _page(d):
                    m = getattr(d, "metadata", None) or {}
                    try:
                        return int(m.get("page", 0) or 0)
                    except Exception:
                        return 0
                docs = sorted(docs, key=_page)

            context = _join_context_within_budget(docs)
            prompt = PROMPT.format(context=context, question=query)

            # Run the async Mistral call in a new event loop (legacy sync path)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        legacy_msgs = [{"role": "user", "content": prompt}]
                        future = pool.submit(asyncio.run, _cs()._call_mistral_async(legacy_msgs))
                        return future.result(timeout=90)
                return loop.run_until_complete(
                    _cs()._call_mistral_async([{"role": "user", "content": prompt}])
                )
            except Exception:
                return _best_chunk_fallback(query, docs)

    return _QARunnable()

