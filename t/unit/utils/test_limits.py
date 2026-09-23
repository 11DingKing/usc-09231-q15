from __future__ import annotations

import pytest

from kombu.utils import limits
from kombu.utils.limits import TokenBucket


class FakeClock:
    """Deterministic stand-in for ``time.monotonic``."""

    def __init__(self, start=0.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def set(self, value):
        self.now = float(value)


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(limits, 'monotonic', fake)
    return fake


def test_bucket_starts_full(clock):
    bucket = TokenBucket(fill_rate=1, capacity=3)
    assert bucket._get_tokens() == 3.0
    assert bucket.timestamp == clock.now


def test_full_bucket_advances_timestamp(clock):
    # Regression: a full bucket skipped the clock update entirely, so the
    # timestamp stayed at the last *partial* state forever.
    bucket = TokenBucket(fill_rate=1, capacity=1)
    clock.advance(10)

    assert bucket.can_consume() is True
    assert bucket.timestamp == 10.0

    # A second call at the same instant must not be refilled for free using
    # the stale t=0 timestamp.
    assert bucket.can_consume() is False
    assert bucket._get_tokens() == 0.0


def test_continuous_high_frequency_calls_drain_exact_capacity(clock):
    capacity = 5
    bucket = TokenBucket(fill_rate=1, capacity=capacity)

    results = [bucket.can_consume() for _ in range(capacity * 4)]

    assert results.count(True) == capacity
    assert results.count(False) == capacity * 3
    assert bucket.timestamp == clock.now


def test_discrete_time_advancement_refills_partial_bucket(clock):
    bucket = TokenBucket(fill_rate=1, capacity=10)
    assert bucket.can_consume(5)
    assert bucket._get_tokens() == 5.0

    clock.advance(3)
    assert bucket._get_tokens() == pytest.approx(8.0)

    clock.advance(3)
    # Refill is capped at capacity even though 11 seconds elapsed since
    # the bucket was drained to 5.
    assert bucket._get_tokens() == pytest.approx(10.0)
    assert bucket.timestamp == 6.0


def test_refill_resumes_after_full_idle_period(clock):
    # Time spent at full capacity must be consumed by the advancing
    # timestamp, not banked as a free refill for the next consumer.
    bucket = TokenBucket(fill_rate=2, capacity=4)

    clock.advance(100)          # idle while full
    assert bucket.can_consume(4)
    assert bucket.timestamp == 100.0

    clock.advance(1)            # only 2 of the 4 tokens back
    assert bucket.can_consume(2) is True
    assert bucket.can_consume(1) is False


def test_partial_consume_is_all_or_nothing(clock):
    bucket = TokenBucket(fill_rate=1, capacity=5)
    bucket.can_consume(4)

    clock.advance(0.5)          # 1.5 tokens available
    assert bucket.can_consume(2) is False
    # Failed request consumes nothing.
    assert bucket._get_tokens() == pytest.approx(1.5)


def test_clock_going_backwards_does_not_remove_tokens(clock):
    bucket = TokenBucket(fill_rate=1, capacity=10)
    clock.advance(100)
    assert bucket.can_consume(5)
    assert bucket._get_tokens() == 5.0

    clock.set(50)               # monotonic clock jumps backwards
    assert bucket._get_tokens() == pytest.approx(5.0)
    assert bucket.timestamp == 50.0

    clock.advance(2)
    assert bucket._get_tokens() == pytest.approx(7.0)


def test_expected_time(clock):
    bucket = TokenBucket(fill_rate=2, capacity=10)
    assert bucket.can_consume(9)
    assert bucket.expected_time(1) == pytest.approx(0.0)

    clock.advance(1)            # 2 tokens refilled -> 3 held
    # Need 10 tokens: 7 short at 2 tokens/second.
    assert bucket.expected_time(10) == pytest.approx(3.5)
