"""Run a FIFO connection pool in event time and sample it on a fixed clock."""

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from random import Random

SAMPLE_MS = 100


@dataclass(frozen=True)
class Reading:
    time_ms: int
    busy: int
    waiting: int


@dataclass(frozen=True)
class WaitEpisode:
    start_ms: int
    end_ms: int | None  # None means the queue was still nonempty at run end.


@dataclass(frozen=True)
class Run:
    seed: int
    duration_ms: int
    capacity: int
    readings: tuple[Reading, ...]
    wait_arrivals_ms: tuple[int, ...]
    episodes: tuple[WaitEpisode, ...]


def arrivals(rng: Random, duration_ms: int):
    """Yield (arrival time, connection hold time) in milliseconds."""
    start_ms = 0
    rate = 7.0
    end_ms = rng.randint(8_000, 22_000)
    while start_ms < duration_ms:
        time_ms = float(start_ms)
        while True:
            time_ms += rng.expovariate(rate / 1000)
            if time_ms >= end_ms or time_ms >= duration_ms:
                break
            hold_ms = max(1, round(rng.expovariate(1 / 850)))
            yield int(time_ms), hold_ms

        start_ms = end_ms
        if rate == 7.0:
            rate = rng.uniform(11.0, 21.0)
            end_ms += rng.randint(2_000, 7_000)
        else:
            rate = 7.0
            end_ms += rng.randint(12_000, 32_000)


def simulate(seed: int, duration_ms: int = 600_000, capacity: int = 12) -> Run:
    if duration_ms <= 0 or duration_ms % SAMPLE_MS or capacity <= 0:
        raise ValueError("Use a positive capacity and duration divisible by 100 ms.")

    requests = iter(arrivals(Random(seed), duration_ms))
    incoming = next(requests, None)
    releases: list[int] = []
    queue: deque[int] = deque()
    wait_arrivals = []
    episodes = []
    episode_start = None
    readings = []

    for sample_ms in range(0, duration_ms + 1, SAMPLE_MS):
        while True:
            arrival_ms = incoming[0] if incoming else duration_ms + 1
            release_ms = releases[0] if releases else duration_ms + 1
            if min(arrival_ms, release_ms) > sample_ms:
                break
            if release_ms <= arrival_ms:
                heappop(releases)
                if queue:
                    heappush(releases, release_ms + queue.popleft())
                    if not queue:
                        episodes.append(WaitEpisode(episode_start, release_ms))
                        episode_start = None
            else:
                _, hold_ms = incoming
                if len(releases) < capacity:
                    heappush(releases, arrival_ms + hold_ms)
                else:
                    if not queue:
                        episode_start = arrival_ms
                    queue.append(hold_ms)
                    wait_arrivals.append(arrival_ms)
                incoming = next(requests, None)
        readings.append(Reading(sample_ms, len(releases), len(queue)))

    if episode_start is not None:
        episodes.append(WaitEpisode(episode_start, None))
    return Run(seed, duration_ms, capacity, tuple(readings),
               tuple(wait_arrivals), tuple(episodes))
