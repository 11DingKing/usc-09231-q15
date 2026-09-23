from __future__ import annotations

import pytest

from kombu.utils.limits import TokenBucket


class FakeClock:
    """Deterministic stand-in for ``time.monotonic``."""

    def __init__(self, current=0.0):
        self.current = float(current)

    def __call__(self):
        return self.current

    def advance(self, seconds):
        self.current += seconds
        return self.current


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr('kombu.utils.limits.monotonic', fake)
    return fake


class test_TokenBucket:

    def test_full_bucket_keeps_timestamp_current(self, clock):
        # Regression test: while the bucket sat at capacity the timestamp
        # used to never advance, so a refill after an idle period was computed
        # against a clock reading from before that idle period.
        bucket = TokenBucket(fill_rate=1, capacity=10)
        for current in (10.0, 50.0, 100.0):
            clock.current = current
            assert bucket.expected_time(1) == 0.0
            assert bucket.timestamp == current

        # Consume one token at t=100 and make another call *without any wall
        # clock time passing*: no refill may have been earned, so waiting a
        # full second before the bucket is full again must be reported.
        clock.current = 100.0
        assert bucket.can_consume(1)
        assert bucket.expected_time(10) == pytest.approx(1.0)

        # The remaining capacity is still available as a burst and not a
        # single token more: long idle time at capacity must not be paid out.
        assert bucket.can_consume(9)
        assert not bucket.can_consume(1)

    def test_burst_after_long_idle_is_limited_to_capacity(self, clock):
        bucket = TokenBucket(fill_rate=1, capacity=10)
        clock.advance(1_000.0)

        consumed = 0
        for _ in range(15):
            if bucket.can_consume(1):
                consumed += 1
            else:
                break
        assert consumed == 10
        assert not bucket.can_consume(1)

    def test_discrete_refill_advances_in_real_time(self, clock):
        bucket = TokenBucket(fill_rate=2, capacity=10)
        assert bucket.can_consume(10)
        assert not bucket.can_consume(1)

        clock.advance(1.0)  # 2 tokens earned
        assert bucket.expected_time(2) == pytest.approx(0.0)
        assert bucket.can_consume(2)
        assert not bucket.can_consume(1)

        clock.advance(1.5)  # 3 more tokens earned
        assert bucket.can_consume(3)
        assert not bucket.can_consume(1)

        clock.advance(0.25)  # only half a token earned
        assert not bucket.can_consume(1)
        assert bucket.expected_time(1) == pytest.approx(0.25)

    def test_repeated_calls_without_elapsed_time(self, clock):
        # High-frequency calls made in the same clock instant must not invent
        # tokens and must keep baselining the timestamp.
        bucket = TokenBucket(fill_rate=3, capacity=5)
        assert bucket.can_consume(5)

        clock.advance(1.0)  # exactly 3 tokens recovered

        decisions = [bucket.can_consume(1) for _ in range(5)]
        assert decisions == [True, True, True, False, False]
        assert bucket.timestamp == clock.current
        assert bucket.expected_time(1) == pytest.approx(1 / 3)

    def test_timestamp_advances_when_consumption_is_denied(self, clock):
        # A failed consume still observes the clock; otherwise the next call
        # would refill against a stale timestamp.
        bucket = TokenBucket(fill_rate=1, capacity=2)
        assert bucket.can_consume(2)

        clock.advance(0.5)
        assert not bucket.can_consume(1)
        assert bucket.timestamp == pytest.approx(0.5)

        clock.advance(0.5)
        assert bucket.can_consume(1)

    def test_clock_rollback_does_not_drain_tokens(self, clock):
        bucket = TokenBucket(fill_rate=1, capacity=10)
        assert bucket.can_consume(3)  # 7 tokens left at t=0

        clock.current = -5.0
        assert bucket._get_tokens() == pytest.approx(7)
        assert bucket.timestamp == -5.0
        assert bucket.expected_time(10) == pytest.approx(3.0)

        # Forward progress after the backwards jump refills normally.
        clock.current = -4.0
        assert bucket._get_tokens() == pytest.approx(8)

    def test_clock_rollback_while_bucket_is_full(self, clock):
        bucket = TokenBucket(fill_rate=1, capacity=4)
        clock.advance(100.0)
        assert bucket._get_tokens() == pytest.approx(4)

        clock.current = 50.0
        assert bucket._get_tokens() == pytest.approx(4)
        assert bucket.timestamp == 50.0

        clock.current = 51.0
        assert bucket.can_consume(1)

    def test_failed_consumption_takes_no_tokens(self, clock):
        bucket = TokenBucket(fill_rate=1, capacity=5)
        assert not bucket.can_consume(6)
        assert bucket._tokens == pytest.approx(5)
        assert not bucket.can_consume(50)
        assert bucket._tokens == pytest.approx(5)

    def test_full_bucket_reports_zero_wait(self, clock):
        bucket = TokenBucket(fill_rate=2, capacity=3)
        clock.advance(42.0)
        assert bucket.expected_time(3) == 0.0
        assert bucket.timestamp == 42.0
