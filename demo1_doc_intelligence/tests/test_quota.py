"""The cost guard, tested without a browser and without an API key.

The deployed app answers with the maintainer's free Groq key, so these limits
are the difference between "the demo is up" and "one visitor spent the day's
quota in ten minutes". The contract they implement is written in config.py.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
import quota  # noqa: E402

NOW = datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc)


def limits(monkeypatch, *, session=8, day=25, gap=4.0, enabled=True):
    monkeypatch.setattr(config, "MAX_ANSWERS_PER_SESSION", session)
    monkeypatch.setattr(config, "MAX_ANSWERS_PER_DAY", day)
    monkeypatch.setattr(config, "MIN_SECONDS_BETWEEN_ANSWERS", gap)
    monkeypatch.setattr(config, "QUOTA_GUARD", enabled)


def test_a_visitor_cannot_spend_the_whole_day(tmp_path, monkeypatch):
    limits(monkeypatch, session=2, day=100)
    store = tmp_path / "quota.json"

    assert quota.reserve(0, None, NOW, store).ok
    assert quota.reserve(1, None, NOW, store).ok

    refused = quota.reserve(2, None, NOW, store)
    assert not refused.ok and refused.reason == "session"
    assert "per visit" in refused.message
    assert quota.used_today(store, NOW) == 2, "a refusal must not spend quota"


def test_the_day_has_a_ceiling_and_it_resets(tmp_path, monkeypatch):
    limits(monkeypatch, session=99, day=3)
    store = tmp_path / "quota.json"

    for _ in range(3):
        assert quota.reserve(0, None, NOW, store).ok

    refused = quota.reserve(0, None, NOW, store)
    assert not refused.ok and refused.reason == "daily"
    assert "resets in" in refused.message
    assert quota.used_today(store, NOW) == 3

    tomorrow = NOW + timedelta(days=1)
    assert quota.used_today(store, tomorrow) == 0, "a new day starts clean"
    assert quota.reserve(0, None, tomorrow, store).ok


def test_a_held_down_enter_key_is_not_a_request_loop(tmp_path, monkeypatch):
    limits(monkeypatch, session=99, day=99, gap=10.0)
    store = tmp_path / "quota.json"
    last = NOW.timestamp()

    waiting = quota.reserve(0, last, NOW + timedelta(seconds=3), store)
    assert not waiting.ok and waiting.reason == "gap"
    assert 6.9 <= waiting.wait_seconds <= 7.1
    assert quota.used_today(store, NOW) == 0

    assert quota.reserve(0, last, NOW + timedelta(seconds=11), store).ok


def test_the_guard_can_be_switched_off_for_local_work(tmp_path, monkeypatch):
    limits(monkeypatch, session=1, day=1, enabled=False)
    store = tmp_path / "quota.json"

    for _ in range(5):
        assert quota.reserve(9, NOW.timestamp(), NOW, store).ok
    assert not store.exists(), "a disabled guard writes nothing"


def test_a_corrupt_counter_does_not_take_the_demo_down(tmp_path, monkeypatch):
    limits(monkeypatch, session=99, day=2)
    store = tmp_path / "quota.json"
    store.write_text("{ this is not json", encoding="utf-8")

    assert quota.used_today(store, NOW) == 0
    assert quota.reserve(0, None, NOW, store).ok
    assert quota.used_today(store, NOW) == 1


def test_remaining_today_is_reported_for_the_log(tmp_path, monkeypatch):
    limits(monkeypatch, session=99, day=5)
    store = tmp_path / "quota.json"
    assert quota.reserve(0, None, NOW, store).remaining_today == 4
    assert quota.check(0, None, NOW, store).remaining_today == 4
