"""Optional board-local SQLite workload that publishes measured connection counts."""

import asyncio
import logging
import random
import sqlite3
import uuid

from .simulator import SAMPLE_MS

CAPACITY = 4
SOURCE = "Board SQLite demo (generated traffic)"


async def _run(publish, stopped):
    loop = asyncio.get_running_loop()
    connections = [sqlite3.connect(":memory:") for _ in range(CAPACITY)]
    pool = asyncio.Queue()
    for connection in connections:
        connection.execute("CREATE TABLE readings (value INTEGER)")
        connection.execute("INSERT INTO readings VALUES (1)")
        connection.commit()
        pool.put_nowait(connection)
    waiting = 0

    async def query(hold):
        nonlocal waiting
        waiting += 1
        try:
            connection = await pool.get()
        finally:
            waiting -= 1
        try:
            connection.execute("BEGIN")
            connection.execute("SELECT value FROM readings").fetchone()
            await asyncio.sleep(hold)
            connection.commit()
        finally:
            pool.put_nowait(connection)

    tasks = set()
    rng = random.Random()
    started = loop.time()

    async def traffic():
        while not stopped.is_set():
            # A bounded alternating load exercises both queue regimes.
            rate = 32 if (loop.time()-started) % 40 < 14 else 9
            await asyncio.sleep(rng.expovariate(rate))
            if len(tasks) < 128:
                task = asyncio.create_task(query(rng.uniform(.08, .24)))
                tasks.add(task)
                task.add_done_callback(tasks.discard)

    generator = asyncio.create_task(traffic())
    session = str(uuid.uuid4())
    last_tick = None
    try:
        while not stopped.is_set():
            tick = int((loop.time()-started)*1000) // SAMPLE_MS
            if last_tick is not None and tick == last_tick:
                await asyncio.sleep(max(0, started+(last_tick+1)*SAMPLE_MS/1000-loop.time()))
                continue
            sample = dict(time_ms=tick*SAMPLE_MS, busy=CAPACITY-pool.qsize(), waiting=waiting)
            payload = dict(origin="demo", source=SOURCE, capacity=CAPACITY, session=session,
                           readings=[sample])
            try:
                await asyncio.to_thread(publish, payload)
            except ValueError as error:
                # Another source owns the board, or this producer missed a tick.
                # End only our session, so a real producer keeps running.
                try:
                    await asyncio.to_thread(publish, {"session":session, "stop":True})
                except ValueError:
                    pass
                logging.debug("SQLite demo waiting to publish: %s", error)
                session = str(uuid.uuid4())
                await asyncio.sleep(1)
            last_tick = tick
    finally:
        generator.cancel()
        outstanding = list(tasks)
        for task in outstanding:
            task.cancel()
        await asyncio.gather(generator, *outstanding, return_exceptions=True)
        for connection in connections:
            connection.close()


def run_demo(publish, stopped):
    asyncio.run(_run(publish, stopped))
