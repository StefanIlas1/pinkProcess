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
- `malformed.csv`: Rows with missing and extra columns. Importing it should fail with a clear message.
- `large-50k.csv`: A long load series at the import row limit, for checking speed.

Every file is under about 200 KB except `large-50k.csv`, which sits at the 50,000 row import limit.
