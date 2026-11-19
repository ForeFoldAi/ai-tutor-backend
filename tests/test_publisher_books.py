"""Publisher books: classes see only their chosen book; default pool unchanged."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import app.modules.school_admin  # noqa: F401  load before enrollment (existing import cycle)
from app.modules.catalog.publisher_books import filter_uploads_for_book
from app.modules.student_learning.enrollment import compute_scope_key


def _uploads(*ids: int):
    return [SimpleNamespace(id=i) for i in ids]


def test_default_pool_hides_publisher_books():
    # 1, 2 = legacy/default chapters; 3, 4 = S.Chand (book 50); 5 = another publisher (book 60)
    book_of = {3: 50, 4: 50, 5: 60}
    ups = _uploads(1, 2, 3, 4, 5)
    assert [u.id for u in filter_uploads_for_book(ups, book_of, None)] == [1, 2]
    assert [u.id for u in filter_uploads_for_book(ups, book_of, 50)] == [3, 4]
    assert [u.id for u in filter_uploads_for_book(ups, book_of, 60)] == [5]


def test_scope_key_unchanged_without_book_choice():
    cls = SimpleNamespace(id=12)
    assert compute_scope_key(cls, ["Math", "Science"]) == "12:math,science"
    assert compute_scope_key(cls, ["Math", "Science"], {}) == "12:math,science"
    # Book chosen for a subject not in the class doesn't change the key either.
    assert compute_scope_key(cls, ["Math"], {"science": 7}) == "12:math"


def test_scope_key_changes_with_book_choice():
    cls = SimpleNamespace(id=12)
    a = compute_scope_key(cls, ["Math", "Science"], {"science": 7})
    b = compute_scope_key(cls, ["Math", "Science"], {"science": 8})
    assert a == "12:math,science|books:science=7"
    assert a != b


def test_other_chapter_hint_stays_in_same_book(monkeypatch):
    import app.core.database as database
    import app.modules.catalog.publisher_books as pb
    from app.services.chapter_scope import _same_book_only

    monkeypatch.setattr(database, "SessionLocal", lambda: nullcontext(None))
    monkeypatch.setattr(pb, "publisher_book_of", lambda _db, _ids: {3: 50, 4: 50})
    per_upload = {"2": ["d"], "4": ["d"], "9": ["d"]}
    # Student on default book (chapter 1): only default-pool chapters suggested.
    assert set(_same_book_only(per_upload, {"1"})) == {"2", "9"}
    # Student on S.Chand (chapter 3): only S.Chand chapters suggested.
    assert set(_same_book_only(per_upload, {"3"})) == {"4"}
