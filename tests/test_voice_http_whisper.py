"""Tests for voice HTTP history and Whisper STT helpers."""

from unittest.mock import MagicMock, patch

from app.services.voice_stt_postprocess import postprocess_voice_transcript
from app.voice_api import ChapterVoiceRequest, ConversationTurn, _normalize_history


def test_normalize_history_caps_turns():
    turns = [
        ConversationTurn(role="user", content=f"q{i}")
        for i in range(20)
    ]
    out = _normalize_history(turns)
    assert len(out) == 14
    assert out[0]["content"] == "q6"


def test_chapter_voice_request_accepts_history():
    req = ChapterVoiceRequest(
        message="hello",
        tutor_state="CHECKING_UNDERSTANDING",
        conversation_history=[
            ConversationTurn(role="assistant", content="What is 2+2?"),
            ConversationTurn(role="user", content="four"),
        ],
    )
    assert req.tutor_state == "CHECKING_UNDERSTANDING"
    assert len(req.conversation_history) == 2


@patch("app.services.voice_whisper_stt._get_model")
def test_whisper_transcribe_sync(mock_get_model):
    from app.services.voice_whisper_stt import _transcribe_sync

    seg = MagicMock()
    seg.text = " x squared equals four "
    seg.avg_logprob = -0.3
    mock_get_model.return_value.transcribe.return_value = ([seg], None)

    with patch("app.services.voice_whisper_stt.tempfile.NamedTemporaryFile") as tmp_cls:
        tmp = MagicMock()
        tmp.__enter__.return_value = tmp
        tmp_cls.return_value = tmp
        out = _transcribe_sync(b"\x00" * 512, language="en", subject_name="Mathematics")

    assert "x squared" in str(out["transcript"])


def test_transcript_likely_echo():
    from app.services.voice_stt_postprocess import transcript_likely_echo

    assert transcript_likely_echo(
        "photosynthesis is the process plants use",
        "Photosynthesis is the process plants use to make food",
    )
    assert not transcript_likely_echo("what is mitochondria", "photosynthesis is the process")


def test_postprocess_after_whisper_math():
    assert "photosynthesis" in postprocess_voice_transcript(
        "photo synthesis", subject_name="Science"
    )
