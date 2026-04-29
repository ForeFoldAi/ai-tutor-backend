from typing import Any

import requests
from langchain_core.prompts import PromptTemplate

from app.config import (
    MISTRAL_API_KEY,
    MISTRAL_MAX_TOKENS,
    MISTRAL_MODEL,
    MISTRAL_TEMPERATURE,
    RETRIEVAL_K,
)


prompt_template = """
You are an academic assistant.

Use ONLY the provided context to answer.
If the answer is not in the document, say:
"The answer is not found in the document."

Context:
{context}

Question:
{question}

Answer clearly:
""".strip()

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def _ensure_mistral_config() -> None:
    if not MISTRAL_API_KEY:
        # main.py catches FileNotFoundError to switch to a doc-only fallback.
        raise FileNotFoundError("Missing MISTRAL_API_KEY env var. Set it to use Mistral for /chat.")


def _call_mistral(prompt: str) -> str:
    _ensure_mistral_config()

    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MISTRAL_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MISTRAL_MAX_TOKENS,
            "temperature": MISTRAL_TEMPERATURE,
        },
        timeout=90,
    )
    resp.raise_for_status()

    payload: dict[str, Any] = resp.json()
    text = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    return text or "The answer is not found in the document."


def get_qa_chain(vectorstore):
    """
    Keeps backward compatibility with main.py's old usage:
        qa_chain = get_qa_chain(vectorstore)
        answer = qa_chain.run(query)
    """

    class _QARunnable:
        def run(self, query: str) -> str:
            retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
            docs = retriever.invoke(query)

            # Keep only non-empty text chunks.
            parts = []
            for d in docs:
                t = (getattr(d, "page_content", "") or "").strip()
                if t:
                    parts.append(t)
            context = "\n\n".join(parts)

            prompt = PROMPT.format(context=context, question=query)
            return _call_mistral(prompt)

    return _QARunnable()
