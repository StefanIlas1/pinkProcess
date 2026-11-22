from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
POOL_CAPACITY = 12
POOL_ROWS = 600


def poisson(rng, mean):
    limit, product, count = math.exp(-mean), rng.random(), 0
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def pool_series(seed, load):
    rng = Random(seed)
    busy, waiting = 0, 0
    rows = []
    for index in range(POOL_ROWS):
        arrivals = poisson(rng, max(0.0, load(index, rng)))
        released = sum(1 for _ in range(busy) if rng.random() < 0.08)
        busy -= released
        waiting += arrivals
        start = min(waiting, POOL_CAPACITY - busy)
        waiting -= start
        busy += start
        rows.append((index * 100, busy, waiting))
    return rows


def write_pool(name, rows):
    lines = ["time_ms,busy,waiting"] + [f"{t},{b},{w}" for t, b, w in rows]
    (HERE / name).write_text("\n".join(lines) + "\n")


def stamp(index, step_seconds):
    moment = datetime(2026, 11, 4, 8, 0, 0) + timedelta(seconds=index * step_seconds)
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def write_table(name, header, rows):
    lines = [",".join(header)] + [",".join(str(value) for value in row) for row in rows]
    (HERE / name).write_text("\n".join(lines) + "\n")


def api_latency():
    rng = Random(21)
    rows = []
    for index in range(720):
        burst = 1 if 300 <= index < 340 or 560 <= index < 575 else 0
        p50 = 90 + 12 * math.sin(index / 40) + rng.uniform(-6, 6) + burst * 40
        p95 = p50 * (2.1 + rng.uniform(0, 0.4)) + burst * 220
        requests = int(900 + 120 * math.sin(index / 90) + rng.uniform(-30, 30))
        errors = int(rng.uniform(0, 3) + burst * rng.uniform(20, 60))
        rows.append((stamp(index, 60), round(p50, 1), round(p95, 1), requests, errors))
    write_table("api-latency.csv", ["timestamp", "p50_ms", "p95_ms", "requests", "errors"], rows)


def server_thermal():
    rng = Random(22)
    rows = []
    temp = 48.0
    for index in range(600):
        heavy = 200 <= index < 380
        cpu = 35 + (50 if heavy else 0) + rng.uniform(-5, 5)
        throttled = temp > 86
        if throttled:
            cpu = min(cpu, 45 + rng.uniform(-4, 4))
        temp += (0.18 if heavy else -0.12) * (1 if not throttled else 0.2) + rng.uniform(-0.3, 0.3)
        temp = max(42.0, min(temp, 92.0))
        memory = 52 + index / 60 + rng.uniform(-1.5, 1.5)
        rows.append((stamp(index, 10), round(cpu, 1), round(memory, 1), round(temp, 1)))
    write_table("server-thermal.csv", ["timestamp", "cpu_pct", "memory_pct", "temp_c"], rows)


def disk_io():
    rng = Random(23)
    rows = []
    for index in range(600):
        backup = 250 <= index < 330
        read = 40 + rng.uniform(-8, 8) + (180 if backup else 0)
        write = 25 + rng.uniform(-6, 6) + (60 if backup else 0)
        depth = max(0, round(2 + (read + write) / 60 + rng.uniform(-1, 1)))
        rows.append((stamp(index, 5), round(read, 1), round(write, 1), depth))
    write_table("disk-io.csv", ["timestamp", "read_mb_s", "write_mb_s", "queue_depth"], rows)


def network_loss():
    rng = Random(24)
    rows = []
    for index in range(600):
        bad = 120 <= index < 170 or 430 <= index < 450
        loss = max(0.0, rng.uniform(0, 0.4) + (rng.uniform(4, 12) if bad else 0))
        rtt = 18 + rng.uniform(-3, 3) + (loss * 6 if bad else 0)
        rows.append((stamp(index, 2), round(loss, 2), round(rtt, 1)))
    write_table("network-loss.csv", ["timestamp", "packet_loss_pct", "rtt_ms"], rows)


def sales():
    rng = Random(25)
    rows = []
    for day in range(365):
        weekly = 1 + 0.25 * math.sin(2 * math.pi * (day % 7) / 7)
        yearly = 1 + 0.35 * math.sin(2 * math.pi * (day - 80) / 365) + (0.9 if 325 <= day <= 330 else 0)
        orders = round(120 * weekly * yearly + rng.uniform(-8, 8))
        rows.append(((date(2026, 1, 1) + timedelta(days=day)).isoformat(), orders, round(orders * (18 + rng.uniform(-1.2, 1.2)), 2)))
    write_table("sales-seasonal.csv", ["date", "orders", "revenue"], rows)


def iot_drift():
    rng = Random(26)
    rows = []
    for index in range(800):
        sensor_a = 20.0 + rng.uniform(-0.15, 0.15) + index * 0.004
        sensor_b = 20.0 + rng.uniform(-0.15, 0.15) + (index * 0.012 if index > 400 else 0)
        rows.append((stamp(index, 30), round(sensor_a, 3), round(sensor_b, 3)))
    write_table("iot-drift.csv", ["timestamp", "sensor_a_c", "sensor_b_c"], rows)


def log_events():
    rng = Random(27)
    lines = []
    for index in range(1500):
        incident = 700 <= index < 760
        status = rng.choice([500, 502, 503]) if incident and rng.random() < 0.5 else rng.choice([200, 200, 200, 201, 204, 404])
        latency = rng.uniform(20, 120) + (rng.uniform(300, 900) if incident else 0)
        lines.append(json.dumps(dict(ts=index * 1000, service=rng.choice(["api", "auth", "search"]), status=status,
                                      latency_ms=round(latency, 1), bytes=int(rng.uniform(300, 9000)))))
    (HERE / "log-events.jsonl").write_text("\n".join(lines) + "\n")


def malformed():
    rng = Random(28)
    lines = ["time_ms,busy,waiting"]
    for index in range(80):
        row = f"{index * 100},{rng.randint(0, 12)},{rng.randint(0, 6)}"
        if index in (12, 40):
            row = f"{index * 100},{rng.randint(0, 12)}"
        if index == 55:
            row = f"{index * 100},many,none,extra"
        lines.append(row)
    (HERE / "malformed.csv").write_text("\n".join(lines) + "\n")


def large():
    rng = Random(29)
    rows = []
    level = 20.0
    for index in range(50_000):
        level += rng.uniform(-0.6, 0.6) + (0.05 if index % 5000 < 400 else -0.002)
        level = max(0.0, min(level, 60.0))
        rows.append(f"{index * 100},{round(level)}")
    (HERE / "large-50k.csv").write_text("time_ms,load\n" + "\n".join(rows) + "\n")


def calm(index, rng):
    return 0.3


def slow_build(index, rng):
    return 0.3 + index / 800


def sudden_spike(index, rng):
    return 0.3 + (1.7 if 300 <= index < 370 else 0)


def recovery(index, rng):
    return 1.4 if index < 200 else max(0.2, 1.4 - (index - 200) / 60)


def oscillating(index, rng):
    return 0.6 + 0.5 * math.sin(index / 28)


MANIFEST = [
    ("pool-calm.csv", "Calm connection pool", "A pool that never queues. Checks that a quiet system reads as quiet."),
    ("pool-slow-build.csv", "Pool, slow build", "Arrivals creep up until the queue starts to grow late in the run."),
    ("pool-sudden-spike.csv", "Pool, sudden spike", "A short burst of traffic fills the pool and queues requests."),
    ("pool-recovery.csv", "Pool, recovery", "A long backlog that drains to zero before the end."),
    ("pool-oscillating.csv", "Pool, oscillating", "Load rises and falls in waves, so the queue comes and goes."),
    ("api-latency.csv", "API latency", "p50 and p95 latency, request counts and two error bursts."),
    ("server-thermal.csv", "Server thermal throttle", "CPU, memory and temperature with a heat event that triggers throttling."),
    ("disk-io.csv", "Disk IO", "Read and write throughput with a backup window that deepens the queue."),
    ("network-loss.csv", "Network packet loss", "Packet loss and round trip time with two bad stretches."),
    ("sales-seasonal.csv", "Sales with seasonality", "A year of daily orders and revenue with weekly and yearly cycles and a sale."),
    ("iot-drift.csv", "IoT sensor drift", "Two temperature sensors, one of which drifts away from the other."),
    ("log-events.jsonl", "Log events (JSON lines)", "Request log events with status, latency and an incident window."),
    ("malformed.csv", "Malformed file", "Rows with missing and extra columns. Importing it should fail with a clear message."),
    ("large-50k.csv", "Large file (50,000 rows)", "A long load series at the import row limit, for checking speed."),
]


def main():
    write_pool("pool-calm.csv", pool_series(11, calm))
    write_pool("pool-slow-build.csv", pool_series(12, slow_build))
    write_pool("pool-sudden-spike.csv", pool_series(13, sudden_spike))
    write_pool("pool-recovery.csv", pool_series(14, recovery))
    write_pool("pool-oscillating.csv", pool_series(15, oscillating))
    api_latency()
    server_thermal()
    disk_io()
    network_loss()
    sales()
    iot_drift()
    log_events()
    malformed()
    large()
    manifest = [dict(file=file, id=Path(file).stem, name=name, description=description) for file, name, description in MANIFEST]
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Test datasets", "",
             "Deterministic sample data for trying the app. Regenerate with `python3 datasets/generate.py`. Pick them from **Import data**, or import the files directly.", ""]
    lines += [f"- `{file}`: {description}" for file, _, description in MANIFEST]
    lines += ["", "Every file is under about 200 KB except `large-50k.csv`, which sits at the 50,000 row import limit."]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
