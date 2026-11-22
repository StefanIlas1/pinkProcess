"""Check queue and FP32 argmax capabilities on an isolated physical overlay."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import signal
import subprocess
from time import perf_counter_ns

import numpy as np

from fpga.check_classifier import board_lock
from lm_reference.flan import Flan
from lm_reference.live_demo import PynqClassifier
from lm_reference.projection_argmax import require_argmax_mmio
from lm_reference.projection_overlay import OverlayProjection, projection_mmio
from pinkprocess.telemetry_device import TelemetryDevice

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bitstream", type=Path)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("state_file", type=Path)
    parser.add_argument("--queue-only", action="store_true")
    parser.add_argument("--captures", type=Path, required=True)
    args = parser.parse_args()
    from pynq import allocate

    unit = "pinkprocess-kit.service"
    with board_lock():
        if subprocess.check_output(["systemctl", "show", "-P", "ActiveState", unit],
                                   text=True).strip() != "inactive":
            raise RuntimeError("The isolated qualification requires an inactive kit service.")
        classifier=PynqClassifier(args.bitstream)
        telemetry=TelemetryDevice(classifier,args.profiles,args.state_file)
        before=telemetry.handle(dict(op="status"))
        rng=np.random.default_rng(391)
        argmax=None if args.queue_only else require_argmax_mmio(classifier.overlay)
        projection=OverlayProjection(projection_mmio(classifier.overlay), allocate, scope="all",
                                     queue=True, argmax_mmio=argmax)
        if projection.native is None:
            raise RuntimeError("The ARM native projection helper did not load.")
        quantized_values = 0
        for depth in (1, 17, 64, 512, 1024):
            for x in (rng.uniform(-5, 5, size=(1, depth)).astype(np.float32),
                      np.zeros((1, depth), dtype=np.float32),
                      np.linspace(-126.5, 126.5, depth, dtype=np.float32).reshape(1, -1)):
                actual = projection.native.quantize_activations(x)
                if actual is None:
                    raise RuntimeError("ARM native quantization unavailable.")
                expected_q, expected_scale = Flan.quantize_activations(x)
                np.testing.assert_array_equal(actual[0], expected_q)
                np.testing.assert_array_equal(actual[1].view(np.uint32), expected_scale.view(np.uint32))
                quantized_values += x.size

        def counters():
            names = ("elapsed_cycles", "mac_feed_cycles", "ddr_read_beats", "ddr_write_beats")
            values = {name: projection.mmio.read(reg) for name, reg in zip(names, (0x2c, 0x30, 0x34, 0x38))}
            assert 0 < values["mac_feed_cycles"] <= values["elapsed_cycles"]
            assert values["ddr_read_beats"] > 0 and values["ddr_write_beats"] > 0
            return values

        def interrupted(*_):
            raise KeyboardInterrupt

        previous=signal.signal(signal.SIGTERM,interrupted)
        begin=perf_counter_ns()
        try:
            queue_scores=0
            queued_counters=[]
            for batch in (9,1):
                x=rng.integers(-128,128,size=(batch,19),dtype=np.int8)
                matrices=tuple((f"qkv.{i}",rng.integers(-128,128,size=(rows,19),dtype=np.int8))
                               for i,rows in enumerate((17,33,66)))
                result=projection.project_many(matrices,x)
                queued_counters.append({"positions": batch, **counters()})
                for (_,weight),actual in zip(matrices,result):
                    np.testing.assert_array_equal(actual,x.astype(np.int32)@weight.astype(np.int32).T)
                    queue_scores+=actual.size
            winner=tie_winner=None
            if argmax is not None:
                depth=512
                x=rng.integers(-128,128,size=(1,depth),dtype=np.int8)
                weight=rng.integers(-128,128,size=(5250,depth),dtype=np.int8)
                row_scales=rng.uniform(.0004,.03,size=len(weight)).astype(np.float32)
                scale=np.float32(.0034)
                raw=x.astype(np.int32)@weight.astype(np.int32).T
                reference=(raw.astype(np.float32)*scale)*row_scales
                winner=projection.head_argmax(weight,x,scale,row_scales)
                assert winner==int(np.argmax(reference)),(winner,int(np.argmax(reference)))
                # Equal maxima at the segment boundary must preserve the earlier token.
                weight[:]=0
                weight[[4095,4096],0]=127
                x[:]=0
                x[0,0]=127
                row_scales[:]=.01
                tie_winner=projection.head_argmax(weight,x,scale,row_scales)
                assert tie_winner==4095,tie_winner
            physical_captures = []
            for mode in ("encoder", "decoder"):
                with np.load(args.captures / f"projection-{mode}.npz", allow_pickle=False) as capture:
                    x, w = capture["activations"], capture["weights"]
                expected = x.astype(np.int32) @ w.astype(np.int32).T
                project_start = perf_counter_ns()
                result = projection.project(mode, w, x)
                measured = perf_counter_ns() - project_start
                np.testing.assert_array_equal(result, expected)
                physical_captures.append({"mode": mode, "shape":{"batch":len(x),"depth":w.shape[1],"rows":len(w)},
                                          "integer_reference_matched": True, "host_roundtrip_ns":measured, **counters()})
            metrics=projection.profile()
        finally:
            if projection.pending:
                # This isolated process owns the board. Reconfigure the PL before
                # releasing retained DMA buffers after an ambiguous timeout.
                try:
                    classifier.overlay.download()
                finally:
                    projection.close()
            else:
                projection.close()
            signal.signal(signal.SIGTERM,previous)
        after=telemetry.handle(dict(op="status"))
        assert before==after and metrics["queue_runs"]==2
        assert classifier.read_row(0)==telemetry.models[telemetry.version].row
        status={"bitstream_sha256":sha256(args.bitstream.read_bytes()).hexdigest(),
                "integer_queue_scores_checked":queue_scores,"queue_matched":True,
                "argmax_matched":None if args.queue_only else True,"argmax_winner":winner,"cross_segment_tie_winner":tie_winner,
                "legacy_telemetry_status_and_reserved_row_matched":True,
                "arm_single_row_values_matched":quantized_values,
                "queued_hardware_counters":queued_counters,"real_capture_counters":physical_captures,
                "metrics":metrics,"qualification_ns":perf_counter_ns()-begin}
        print(json.dumps(status,indent=2))

if __name__=="__main__":
    main()
