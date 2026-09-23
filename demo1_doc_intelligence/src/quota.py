"""Keep a public demo from spending the whole daily quota in ten minutes.

The deployed app answers with the maintainer's Groq key. Anyone with the URL
can spend that daily budget, which is why retrieval.capability_answer() exists
for questions that need no model at all, and why this module exists for the
questions that do.

Three limits, all stated in config.py next to the cost contract:

* per session, so one visitor cannot use the whole day;
* per day, so the demo is still alive for the next visitor;
* a minimum gap between answers, so a held-down Enter key is not a request loop.

The counter lives in a small JSON file because a free Streamlit container has no
shared store and may restart at any time. A restart resets the counter, which is
accepted: the provider's own daily limit is the backstop, and the guard exists
to shape behaviour, not to meter billing.

Pure functions with an injectable path and clock, so the caps are testable
without a browser or an API key.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config


@dataclass(frozen=True)
class Reservation:
    """The answer to "may this question be sent to the model?"."""

    ok: bool
    reason: str = ""          # "" when ok, else "session" | "daily" | "gap"
    message: str = ""
    remaining_today: int = 0
    wait_seconds: float = 0.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _day(now: datetime) -> str:
    return now.date().isoformat()


def _path(path=None) -> Path:
    return Path(path or config.QUOTA_FILE)


def _read(path, now) -> dict:
    """Today's usage. A corrupt or stale file counts as zero, never as an error."""
    try:
        data = json.loads(_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"date": _day(now), "used": 0}
    if data.get("date") != _day(now):
        return {"date": _day(now), "used": 0}
    return {"date": data["date"], "used": int(data.get("used") or 0)}


def _write(path, data) -> None:
    target = _path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(data), encoding="utf-8")
        temp.replace(target)     # atomic enough for one container
    except OSError:
        pass                     # a guard that crashes the app is worse than no guard


def used_today(path=None, now=None) -> int:
    return _read(path, now or _now())["used"]


def seconds_until_reset(now=None) -> float:
    """Until the counter rolls over. Groq resets on rolling windows; the local
    counter uses the calendar day in UTC, so the message says exactly that
    rather than implying a rolling window it does not implement."""
    now = now or _now()
    tomorrow = datetime.combine(
        now.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )
    return max(0.0, (tomorrow - now).total_seconds())


def _format_wait(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes >= 60:
        return f"{minutes // 60}h {minutes % 60}m"
    return f"{minutes}m" if minutes else f"{int(seconds)}s"


def check(session_used=0, last_answer_at=None, now=None, path=None) -> Reservation:
    """May another answer be generated? Does not consume anything."""
    now_dt = now or _now()
    used = _read(path, now_dt)["used"]
    remaining = max(0, config.MAX_ANSWERS_PER_DAY - used)

    if not config.QUOTA_GUARD:
        return Reservation(ok=True, remaining_today=remaining)

    if session_used >= config.MAX_ANSWERS_PER_SESSION:
        return Reservation(
            ok=False,
            reason="session",
            remaining_today=remaining,
            message=(
                f"**Session limit reached.** This demo answers "
                f"{config.MAX_ANSWERS_PER_SESSION} questions per visit, so that "
                "one visitor cannot use the whole day's quota for everyone "
                "else. Reload the page to start a new session, or run the demo "
                "locally with your own Groq key."
            ),
        )

    if remaining <= 0:
        return Reservation(
            ok=False,
            reason="daily",
            remaining_today=0,
            message=(
                f"**Demo quota used for today.** The demo answers "
                f"{config.MAX_ANSWERS_PER_DAY} questions a day on a free API "
                f"key, and that is spent. It resets in "
                f"{_format_wait(seconds_until_reset(now_dt))} (calendar day, "
                "UTC). You can still run the demo locally with your own key."
            ),
        )

    if last_answer_at is not None:
        elapsed = now_dt.timestamp() - last_answer_at
        wait = config.MIN_SECONDS_BETWEEN_ANSWERS - elapsed
        if wait > 0:
            return Reservation(
                ok=False,
                reason="gap",
                remaining_today=remaining,
                wait_seconds=wait,
                message=(
                    f"**One moment.** Please wait {wait:.0f} more seconds "
                    "between questions."
                ),
            )

    return Reservation(ok=True, remaining_today=remaining)


def reserve(session_used=0, last_answer_at=None, now=None, path=None) -> Reservation:
    """check(), and on success count the answer against today's budget.

    The count happens before the model call, so an answer that fails still
    counts: the quota is spent on attempts, and pretending otherwise would let
    a failing key or a hammering script look free.
    """
    now_dt = now or _now()
    result = check(session_used, last_answer_at, now_dt, path)
    if not result.ok or not config.QUOTA_GUARD:
        return result

    data = _read(path, now_dt)
    data["used"] += 1
    _write(path, data)
    return Reservation(
        ok=True,
        remaining_today=max(0, config.MAX_ANSWERS_PER_DAY - data["used"]),
    )
