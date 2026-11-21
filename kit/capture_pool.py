"""Measure waits at a four-connection FIFO pool serving real SQLite reads."""

import argparse
import asyncio
import csv
import json
import random
import sqlite3
import tempfile
import time
from bisect import bisect_right
from collections import deque
from pathlib import Path

CAPACITY = 4
DURATION = 60
SEEDS = (7101, 7102, 7103)


async def capture(folder, database, seed, varied=False):
    rng = random.Random(seed)
    available = [sqlite3.connect(database) for _ in range(CAPACITY)]
    connections = available.copy()
    waiters = deque()
    events, timestamps = [], []
    started = time.monotonic_ns()

    def event(kind, number):
        stamp = (time.monotonic_ns() - started) // 1000
        timestamps.append(stamp)
        events.append((stamp, kind, number, CAPACITY - len(available), len(waiters)))

    async def request(number, hold, key):
        if available:
            conn = available.pop()
            event("acquire", number)
        else:
            future = asyncio.get_running_loop().create_future()
            waiters.append((number, future))
            event("enqueue", number)
            conn = await future
        try:
            conn.execute("BEGIN")
            row = conn.execute("SELECT value FROM items WHERE id=?", (key,)).fetchone()
            if row != (key * 3,):
                raise RuntimeError("Unexpected result from the workload database.")
            # Controlled client dwell while the transaction holds a real connection.
            await asyncio.sleep(hold)
            conn.commit()
        finally:
            if waiters:
                next_number, future = waiters.popleft()
                event("handoff", next_number)
                future.set_result(conn)
            else:
                available.append(conn)
                event("release", number)

    async def submit():
        tasks = []
        # Two bursts per run, separated by sustained quiet and moderate traffic.
        schedule = [(0, 15, 8), (15, 20, 36), (20, 35, 5),
                    (35, 43, 19), (43, 48, 32), (48, DURATION, 6)]
        if varied:
            onset = rng.randint(6, 14)
            stop = onset + rng.randint(12, 25)
            schedule = [(0, onset, 7), (onset, stop, rng.uniform(29, 43)),
                        (stop, 48, rng.uniform(9, 24)), (48, DURATION, 7)]
        for begin, end, rate in schedule:
            due = float(begin)
            while True:
                due += rng.expovariate(rate)
                if due >= end:
                    break
                await asyncio.sleep(max(0, due - (time.monotonic_ns() - started) / 1e9))
                tasks.append(asyncio.create_task(request(len(tasks), rng.uniform(.08,.24), rng.randrange(1000))))
        await asyncio.gather(*tasks)
        return len(tasks)

    folder.mkdir()
    task = asyncio.create_task(submit())
    with (folder / "telemetry.csv").open("w") as output:
        output.write("time_ms,busy,waiting\n")
        for ms in range(0, DURATION * 1000 + 1, 100):
            await asyncio.sleep(max(0, ms / 1000 - (time.monotonic_ns() - started) / 1e9))
            idx = bisect_right(timestamps, ms * 1000) - 1
            busy, waiting = events[idx][3:] if idx >= 0 else (0, 0)
            output.write(f"{ms},{busy},{waiting}\n")
            output.flush()
    count = await task
    for conn in connections:
        conn.close()
    with (folder / "events.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("time_us","event","request_id","busy","waiting"))
        writer.writerows(events)
    print(f"Captured {seed}: {count} transactions", flush=True)


async def main(directory, seeds=SEEDS, varied=False):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "protocol.json").write_text(json.dumps(dict(
        varied=varied, duration_seconds=DURATION, seeds=seeds, capacity=CAPACITY, sample_ms=100,
        forecast_ms=10000, evaluation="Frozen profiles, consecutive warnings collapsed; first 5s warmup, final 10s withheld from predictions.",
        source="Local SQLite; scheduled synthetic client traffic and dwell; measured FIFO lease events."
    ), indent=2))
    with tempfile.TemporaryDirectory() as tmp:
        database = Path(tmp) / "workload.sqlite"
        conn = sqlite3.connect(database)
        conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, value INTEGER)")
        conn.executemany("INSERT INTO items VALUES(?,?)", ((i,i*3) for i in range(1000)))
        conn.commit()
        conn.close()
        for seed in seeds:
            await capture(directory / str(seed), database, seed, varied)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--varied", action="store_true", help="Vary overload onset, duration and rate by seed.")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Choose distinct run seeds.")
    asyncio.run(main(args.directory, args.seeds, args.varied))
