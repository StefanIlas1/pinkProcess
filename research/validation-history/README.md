# Validation history

These captures and scripts document earlier versions of the operator UI and FPGA qualification workflow. Use the [board evidence](../../kit/evidence/) for measurement context and the [research experiments](../experiments/) for recorded test data. The historical scripts retain the UI selectors, dependencies and input paths used at the time of each run.

| Run | What to inspect |
| --- | --- |
| Browser flows with test fixtures | The three `pinkprocess-merge-ui` captures show connected, disconnected and narrow-screen states. The speech capture shows a dictated question and CSV upload. The integration capture exercises the interface after an intentionally failed edit. Their [Playwright scripts](browser-checks/) identify the mock board or injected test responses. |
| Connected board routes | The three `pynq-c6-route` captures show historical question and follow-up flows. The [verification script](browser-checks/pynq-c6-verify-ui.py) checks that each completed answer used accelerated projections and that the JSON export retained the answer. It covers [live](captures/pynq-c6-verified-live.png), [sample](captures/pynq-c6-verified-sample.png) and [uploaded CSV](captures/pynq-c6-verified-csv.png) inputs. |
| Healthy queue | The [earlier result](captures/pinkprocess-open-diagnosis.png) reported an empty queue and a forecast warning. The [later qualification](captures/pynq-process-current-healthy-qualification.png) reported zero requests waiting and showed the selected answer skill. |
| Local analyst | The [model QA driver](browser-checks/pinkprocess-model-qa.py) evaluates a recovered queue, increasing pressure, unavailable causal attribution and the mean queue length. Its expected observations distinguish captured telemetry from generated fixtures. |
| Overlay checks | The [board-side scripts](overlay-checks/) compare signed projection results across tensor shapes, exercise the queue and argmax paths, and check concurrent telemetry ingestion during projection. The HWH checker verifies the 50 MHz candidate design's clock routing, register address ranges and DDR access. |
