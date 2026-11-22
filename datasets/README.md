# Test datasets

Deterministic sample data for trying the app. Regenerate with `python3 datasets/generate.py`. Pick them from **Import data**, or import the files directly.

- `pool-calm.csv`: A pool that never queues. Checks that a quiet system reads as quiet.
- `pool-slow-build.csv`: Arrivals creep up until the queue starts to grow late in the run.
- `pool-sudden-spike.csv`: A short burst of traffic fills the pool and queues requests.
- `pool-recovery.csv`: A long backlog that drains to zero before the end.
- `pool-oscillating.csv`: Load rises and falls in waves, so the queue comes and goes.
- `api-latency.csv`: p50 and p95 latency, request counts and two error bursts.
- `server-thermal.csv`: CPU, memory and temperature with a heat event that triggers throttling.
- `disk-io.csv`: Read and write throughput with a backup window that deepens the queue.
- `network-loss.csv`: Packet loss and round trip time with two bad stretches.
- `sales-seasonal.csv`: A year of daily orders and revenue with weekly and yearly cycles and a sale.
- `iot-drift.csv`: Two temperature sensors, one of which drifts away from the other.
- `log-events.jsonl`: Request log events with status, latency and an incident window.
- `checkout-api-10k.csv`: Request telemetry with endpoint, region, status, latency and database pool wait. A deploy breaks one endpoint in one region for twenty minutes.
- `cold-chain-10k.csv`: Ten refrigerated trucks reporting temperature, door state and compressor current every five minutes. One compressor fails.
- `store-sales-10k.csv`: Hourly orders and revenue by store, region and payment method over ten days. A payment outage hits one region.
- `malformed.csv`: Rows with missing and extra columns. Importing it should fail with a clear message.
- `large-50k.csv`: A long load series at the import row limit, for checking speed.

Every file is under about 200 KB except the three 10,000 row files (under 1 MB) and `large-50k.csv`, which sits at the 50,000 row import limit.

## Planted causes

Each 10,000 row file hides one cause that the analyst should find. Times are the clock times in the file.

- `checkout-api-10k.csv`: After a deploy at 14:10 on 2026-11-18, `/checkout` requests in `eu-west` are slow and fail from 14:12 to 14:32 because `db_pool_wait_ms` jumps to 700 to 2,200 ms; every other endpoint and region stays normal.
- `cold-chain-10k.csv`: The compressor on truck `T07` fails at 14:00 on 2026-11-19 (current drops to about 0.4 A) and returns at 19:00; its `temp_c` climbs from 3 C past the 8 C safe limit from about 16:10 to 19:20. Door openings on truck `T03` are a red herring: temperature there never passes 7 C.
- `store-sales-10k.csv`: A card payment outage in the `north` region (all four stores) cuts card orders and revenue by 96 percent from 12:00 to 17:00 on 2026-10-30; cash orders are unaffected.
