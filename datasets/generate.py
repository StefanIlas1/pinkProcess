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
        room = 20.0 + index * 0.004
        sensor_a = room + rng.uniform(-0.15, 0.15)
        sensor_b = room + rng.uniform(-0.15, 0.15) + (index - 400) * 0.012 * (index > 400)
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


def checkout_api_10k():
    rng = Random(31)
    endpoints = [("/search", 35, 60), ("/product", 20, 50), ("/cart", 15, 70), ("/checkout", 20, 140), ("/login", 10, 90)]
    regions = [("us-east", 45), ("eu-west", 30), ("ap-south", 25)]
    begin = datetime(2026, 11, 18, 12, 0, 0)
    incident_start, incident_end = 7920, 9120
    rows = []
    for index in range(10_000):
        seconds = index * 1.44 + rng.uniform(0, 1)
        endpoint, _, base = rng.choices(endpoints, weights=[weight for _, weight, _ in endpoints])[0]
        region = rng.choices([name for name, _ in regions], weights=[weight for _, weight in regions])[0]
        latency = base * rng.lognormvariate(0, 0.25)
        pool_wait = rng.uniform(1, 12)
        failed = rng.random() < 0.005
        broken = endpoint == "/checkout" and region == "eu-west" and incident_start <= seconds < incident_end
        if broken:
            pool_wait = rng.uniform(700, 2200)
            latency += pool_wait
            failed = rng.random() < 0.45
        if failed:
            status = rng.choice([500, 500, 503])
        else:
            status = 404 if endpoint == "/product" and rng.random() < 0.015 else 200
        rows.append(((begin + timedelta(seconds=int(seconds))).strftime("%Y-%m-%d %H:%M:%S"), endpoint, region, status,
                     round(latency, 1), round(pool_wait, 1)))
    write_table("checkout-api-10k.csv", ["timestamp", "endpoint", "region", "status", "latency_ms", "db_pool_wait_ms"], rows)


def cold_chain_10k():
    rng = Random(32)
    trucks = [f"T{number:02d}" for number in range(1, 11)]
    begin = datetime(2026, 11, 17, 0, 0, 0)
    failure_start, failure_end = 744, 804
    door_runs = {}
    for start in rng.sample(range(20, 960, 40), 14):
        if not failure_start - 30 <= start <= failure_end + 30:
            door_runs[start] = rng.randint(3, 8)
    door_runs[520] = 20
    temperature = {truck: 3.0 for truck in trucks}
    rows = []
    for step in range(1000):
        moment = begin + timedelta(minutes=5 * step)
        hour = moment.hour + moment.minute / 60
        ambient = 18 + 8 * math.sin(2 * math.pi * (hour - 9) / 24) + rng.uniform(-0.5, 0.5)
        for truck in trucks:
            failed = truck == "T07" and failure_start <= step < failure_end
            pulling_down = truck == "T07" and failure_end <= step < failure_end + 10
            opened = (truck == "T03" and any(start <= step < start + length for start, length in door_runs.items())) or rng.random() < 0.002
            amps = 7 + 0.15 * (ambient - 20) + rng.uniform(-0.4, 0.4) + (2 if opened else 0)
            drift = (3.0 - temperature[truck]) * 0.15 + (ambient - 20) * 0.01 + rng.uniform(-0.12, 0.12)
            if failed:
                amps = rng.uniform(0.2, 0.6)
                drift = 0.19 + 0.015 * (ambient - 20) + rng.uniform(-0.05, 0.05)
            if pulling_down:
                amps = 10 + rng.uniform(-0.4, 0.4)
            temperature[truck] = min(14.0, temperature[truck] + drift + (0.4 if opened else 0))
            rows.append((moment.strftime("%Y-%m-%d %H:%M:%S"), truck, round(temperature[truck], 2), int(opened), round(amps, 2), round(ambient, 1)))
    write_table("cold-chain-10k.csv", ["timestamp", "truck_id", "temp_c", "door_open", "compressor_amps", "ambient_c"], rows)


def store_sales_10k():
    rng = Random(33)
    regions = ["north", "south", "east", "west", "central"]
    stores = [(f"{region}-{number}", region, 22 + 3 * number + 2 * index) for index, region in enumerate(regions) for number in (1, 2, 3, 4)]
    payments = [("card", 0.65, 26.0), ("cash", 0.35, 19.0)]
    begin = datetime(2026, 10, 26, 0, 0, 0)
    outage_start, outage_end = 4 * 24 + 12, 4 * 24 + 17
    rows = []
    for hour_index in range(250):
        moment = begin + timedelta(hours=hour_index)
        hour = moment.hour
        daylight = 0.0 if hour < 7 or hour >= 23 else 0.5 + 0.5 * math.sin(math.pi * (hour - 7) / 16)
        weekend = 1.25 if moment.weekday() >= 5 else 1.0
        for store, region, size in stores:
            for method, share, ticket in payments:
                expected = size * share * daylight * weekend
                if region == "north" and method == "card" and outage_start <= hour_index < outage_end:
                    expected *= 0.04
                orders = max(0, round(rng.gauss(expected, math.sqrt(expected) if expected > 0 else 0)))
                revenue = round(orders * ticket * rng.uniform(0.9, 1.1), 2)
                rows.append((moment.strftime("%Y-%m-%d %H:%M:%S"), region, store, method, orders, revenue))
    write_table("store-sales-10k.csv", ["timestamp", "region", "store", "payment_method", "orders", "revenue"], rows)


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
    ("checkout-api-10k.csv", "Checkout API, 10,000 requests", "Request telemetry with endpoint, region, status, latency and database pool wait. A deploy breaks one endpoint in one region for twenty minutes."),
    ("cold-chain-10k.csv", "Cold chain trucks, 10,000 readings", "Ten refrigerated trucks reporting temperature, door state and compressor current every five minutes. One compressor fails."),
    ("store-sales-10k.csv", "Store sales, 10,000 rows", "Hourly orders and revenue by store, region and payment method over ten days. A payment outage hits one region."),
    ("malformed.csv", "Malformed file", "Rows with missing and extra columns. Importing it should fail with a clear message."),
    ("large-50k.csv", "Large file (50,000 rows)", "A long load series at the import row limit, for checking speed."),
]


GROUND_TRUTH = [
    ("checkout-api-10k.csv", "After a deploy at 14:10 on 2026-11-18, `/checkout` requests in `eu-west` are slow and fail from 14:12 to 14:32 because `db_pool_wait_ms` jumps to 700 to 2,200 ms; every other endpoint and region stays normal."),
    ("cold-chain-10k.csv", "The compressor on truck `T07` fails at 14:00 on 2026-11-19 (current drops to about 0.4 A) and returns at 19:00; its `temp_c` climbs from 3 C past the 8 C safe limit from about 16:10 to 19:20. Door openings on truck `T03` are a red herring: temperature there never passes 7 C."),
    ("store-sales-10k.csv", "A card payment outage in the `north` region (all four stores) cuts card orders and revenue by 96 percent from 12:00 to 17:00 on 2026-10-30; cash orders are unaffected."),
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
    checkout_api_10k()
    cold_chain_10k()
    store_sales_10k()
    malformed()
    large()
    manifest = [dict(file=file, id=Path(file).stem, name=name, description=description) for file, name, description in MANIFEST]
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Test datasets", "",
             "Deterministic sample data for trying the app. Regenerate with `python3 datasets/generate.py`. Pick them from **Import data**, or import the files directly.", ""]
    lines += [f"- `{file}`: {description}" for file, _, description in MANIFEST]
    lines += ["", "Every file is under about 200 KB except the three 10,000 row files (under 1 MB) and `large-50k.csv`, which sits at the 50,000 row import limit.",
              "", "## Planted causes", "",
              "Each 10,000 row file hides one cause that the analyst should find. Times are the clock times in the file.", ""]
    lines += [f"- `{file}`: {truth}" for file, truth in GROUND_TRUTH]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
