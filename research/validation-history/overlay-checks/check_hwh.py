import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
include_argmax = "--queue-only" not in sys.argv[2:]
checks = (
    ("classifier", "pinkprocess_classifier", "s_axi_aclk"),
    ("projection", "pinkprocess_projection", "s_axi_aclk"),
    ("projection_argmax", "pinkprocess_argmax", "s_axi_aclk"),
    ("projection_ddr", "smartconnect", "aclk"),
    ("ps7", "processing_system7", "FCLK_CLK0"),
    ("ps7", "processing_system7", "M_AXI_GP0_ACLK"),
    ("ps7", "processing_system7", "S_AXI_HP0_ACLK"),
)
for instance, kind, clock_name in checks:
    if not include_argmax and instance == "projection_argmax":
        continue
    module = root.find(f'.//MODULE[@INSTANCE="{instance}"]')
    assert module is not None and module.get("MODTYPE") == kind, instance
    port = module.find(f'./PORTS/PORT[@NAME="{clock_name}"]')
    assert port is not None and port.get("CLKFREQUENCY") == "50000000", (instance, clock_name)
    assert port.get("SIGNAME") == "ps7_FCLK_CLK0", (instance, clock_name)
    print(f"{instance}/{clock_name} uses the 50 MHz PS fabric clock")

ps = root.find('.//MODULE[@INSTANCE="ps7"]')
ranges = []
for instance in (("classifier", "projection", "projection_argmax") if include_argmax else ("classifier", "projection")):
    mapping = ps.find(f'./MEMORYMAP/MEMRANGE[@INSTANCE="{instance}"]')
    assert mapping is not None and mapping.get("MASTERBUSINTERFACE") == "M_AXI_GP0", instance
    start, end = (int(mapping.get(key), 16) for key in ("BASEVALUE", "HIGHVALUE"))
    assert end >= start
    ranges.append((start, end))
    print(f"{instance} registers {start:#010x}–{end:#010x}")
for previous, following in zip(sorted(ranges), sorted(ranges)[1:]):
    assert previous[1] < following[0], "MMIO ranges overlap"
for instance in (("projection", "projection_argmax") if include_argmax else ("projection",)):
    module = root.find(f'.//MODULE[@INSTANCE="{instance}"]')
    aperture = module.find('./MEMORYMAP/MEMRANGE[@SLAVEBUSINTERFACE="S_AXI_HP0"]')
    assert aperture is not None and aperture.get("INSTANCE") == "ps7", instance
    print(f"{instance} reaches DDR via Zynq HP0")
