# pinkProcess

A deterministic simulator and forecasting baseline for a service with a finite
database connection pool. Requires Python 3.10 or later and uses the standard
library. From the project root, run:

```sh
python3 -m pinkprocess
```

The demo fits a rule on eight 10-minute runs, then prints event-level metrics
and two concrete forecasts from four different 10-minute runs.

## Simulation and labels

- `simulator.py` models 12 connections with a FIFO queue. Arrivals have
  exponential gaps within each traffic regime. Ordinary traffic averages seven
  arrivals per second, and intermittent bursts average 11 to 21. Connection
  holds are sampled independently with a mean of about 850 ms. The simulator
  processes arrivals and releases in millisecond event time, with releases first
  on ties, then records busy and waiting counts every 100 ms. Each run's seed
  controls its traffic and service times.
- `dataset.py` emits an example at time `t` only when nobody is waiting and
  both windows fit within the run. The input contains 50 readings from
  `(t - 5 s, t]`. The label is positive when an arrival joined the queue in
  `(t, t + 10 s]`. Short queues can form and drain between samples, so labels
  come from the exact arrival log. Overlapping windows share data, so training
  and evaluation use disjoint sets of run seeds.

## Baseline and scoring

The score adds current utilization to the change in utilization per second
over the most recent two seconds multiplied by a positive weight in seconds.
The score can exceed 1 and has no probability interpretation.
The training grid chooses the weight and threshold that maximize
`2 × warned / (2 × warned + missed + false alerts)`, breaking ties in favor
of fewer false alerts.

A queue episode starts when an arriving request joins an empty queue and ends
when the queue drains. An alert starts on the first positive prediction in a
streak of consecutive 100 ms predictions. A negative prediction or a skipped
tick breaks the streak. Each alert start is matched to the first episode that
begins within its forecast. An episode gets one warning credit, and its lead
time runs from its earliest matched alert to its first queued arrival. Later
alert starts matched to that episode are reported as repeats. An alert start
with no episode in its forecast is false. A sustained positive streak starts
only one alert, even if a later prediction in the streak has a different
outcome.

The episode count includes starts after the first possible prediction time
(5 s) through run end. Near the end, later episodes have fewer eligible
prediction times because prediction stops 10 seconds before run end. False
alerts per hour use the total simulated duration of the held-out runs.
