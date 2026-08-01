"""ponytail: assert-based self-check for scope key + progress math + agent modes. Run: python -m app.modules.student_learning.check"""

from __future__ import annotations

from types import SimpleNamespace

from app.modules.student_learning.enrollment import compute_scope_key


class _FakeClass:
    def __init__(self, id: int) -> None:
        self.id = id


def main() -> None:
    assert compute_scope_key(None, []) == "none:"
    assert compute_scope_key(_FakeClass(12), ["Math", "Science"]) == "12:math,science"
    assert compute_scope_key(_FakeClass(12), ["Science", "Math"]) == "12:math,science"
    completed, total = 2, 5
    progress = int(round((completed / total) * 100)) if total else 0
    assert progress == 40
    assert int(round((5 / 5) * 100)) == 100

    from app.services.chat_service import _apply_agent_mode, _normalize_agent_mode
    from app.services.conversation_context import ResponseMode

    conv = SimpleNamespace(response_mode=ResponseMode.EXPLANATION)
    assert "PRACTICE" in _apply_agent_mode(conv, "practice").upper()
    assert conv.response_mode == ResponseMode.QUIZ
    assert _normalize_agent_mode("explain") == "explain"
    print("student_learning.check OK")


if __name__ == "__main__":
    main()
