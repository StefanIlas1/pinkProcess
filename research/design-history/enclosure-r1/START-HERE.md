# pinkProcess | PYNQ-Z2 desk enclosure R1
Fit-test release. Units mm. Print the STLs directly. The parametric OpenSCAD source also includes an assembled view. Pink PETG gives the intended colour. CAD-preview.png shows the actual mesh surfaces.

## What to print
| STL | Quantity | Function and starting orientation |
|---|---:|---|
| gauge | 1 before full print | Flat mounting-pattern template for the LCD. Wider pair of pins at top, button tab below. |
| keeper | 4 installed | Flat split collars for the screen pins. Try one on the gauge before printing four. |
| shell | 1 | Face down. Bezel, four LCD pin mounts, embossed-in pinkProcess mark, snap windows and cable exits. |
| lid | 1 | Rear carrier and cover, PCB catches, vent rows and kickstand sockets. Export stands on its long edge. Use a wide brim and support overhangs at the hinge ears and PCB clips. Alternative: orient exterior down and support the lifted face. |
| stand | 1 | Flat, with integrated hinge pins and travel-stop shoulders. |
Installed total: 7 printed pieces in 4 designs; no loose hinge pin or purchased case screws. Body 208 x 166 x 55.2 before exterior hinge/stand, about 66.1 mm maximum closed depth including hinge ears. Recommended bed at least 220 x 220 mm, with 170 mm usable Z for the supplied lid orientation. Check the slicer's printable area, brim and machine exclusion zones.

## Hardware this revision is laid out for
- One bare Waveshare 7inch HDMI LCD (H), SKU 14628, 1024 x 600 capacitive touch. Mount drawing corresponds to the Rev4.1 illustration. External WaveShare factory case, Raspberry Pi DSI screens and model C are outside this fit target. Controller supplied with the H module. No separate ribbon interface or auxiliary daughterboard.
- One TUL PYNQ-Z2 board, nominal 140 x 87, with its original heatsink. The rear cover captures the PCB near its corners. Orient connector faces toward the open cable bays. Assumed PCB thickness 1.6 mm; verify on your board.
- One microSD card with the PYNQ boot image.
- One short flexible HDMI Type A male/male cable, PYNQ HDMI OUT to display HDMI. Target roughly 150–200 mm as a shopping starting point. Measure plug bodies, exit direction and bend radius with the parts on the bench before selecting it. Compact right-angle plugs may help.
- One USB data cable from PYNQ USB host to the screen's Touch input for touch, if enabled. Confirm plug type on display revision. Touch software support is separate from case fit.
- Board USB power cable/source for the PYNQ desk configuration, and a suitable 5 V display USB supply/cable per Waveshare instructions if the touch feed alone cannot power the screen. Size and verify the power arrangement on the bench; no built-in regulator, battery, fan or power distribution board.
- One Ethernet cable from the PYNQ to the local network; Mac app connects over that network. Power and network cables pass through open-sided cable exits in the primary shell. Standard HDMI, USB and Ethernet plug bodies vary. Both bays open when the cover comes off; these are cable exits, not individually located connector cutouts.

## Fit procedure
1. Print gauge and one keeper. Dry-fit the bare screen, with no force. Top pin spacing 156.90, bottom 148.90, row spacing 106.96. The published complete height including button tab is 124.27 and panel width 164.90. Trial pin diameter 2.5. Keeper bore 2.28 with longitudinal slit. The gauge only tests XY registration and collar grip.
2. Before the full shell, measure from the display front to the back surface of the four PCB mounting ears, measure PCB thickness, and locate the active image relative to the ear centres. CAD assumes 5.6 mm front-to-back-of-ear (8.4 minus 2.8), 1.6 mm PCB thickness and a centered 158 x 92 window. If the glass would carry pressure, shift the supports or aperture before printing. For the seated display, the posts and bezel should support the module with no bending.
3. Print in PETG. Starting settings: 0.4 mm nozzle, 0.2 mm layers, 4 walls, 4 top/bottom layers, 20–30% infill. Use the filament maker's temperature range. Inspect slicer preview for small pins and thin clip walls. Remove support carefully. Do not scale one STL to solve a fit issue; edit the relevant dimensions and re-export.
4. With power off, lower the display onto the four shell pins. Press one collar onto each pin just to contact the PCB ear. Fit the PYNQ to the rear cover, component side facing into the case. Check each saddle contacts bare board edge only. Fit the microSD before closing.
5. Connect internal HDMI and optional USB touch, then network and power leads. Dry-close with all plugs seated, checking the glass, heatsink, button tab, cable strain reliefs, and snap tabs. The stacked layout leaves approx. 36.5 mm between the nominal component-side PCB planes for both sets of components and cables. Route leads clear of clips and vents. Release the four lid tongues through the side windows for service.
6. Pinch the stand arms inward slightly and seat the pivot pins into the exterior ears, one side at a time. Orient the long arms toward the bottom and the short stop shoulders above the pivot. Check deployment, stops, desk stability and cable pull before relying on the stand.
7. Operate the populated unit under representative load and compare open/closed temperatures. Passive vent openings are present. Thermal performance and printed durability require physical validation.

## What is verified
OpenSCAD full geometry render and STL export completed. All five files checked for one connected closed surface, each edge shared by two oppositely oriented triangles and positive enclosed volume. No physical print, slicer-specific toolpath, exact connector collision check or thermal measurement has been performed. Board clips, stand pivots and cable selection are provisional until tested on hardware.

## Dimension references
- Waveshare SKU 14628: https://www.waveshare.com/7inch-HDMI-LCD-H.htm
- Dimension drawing and connector view, page 11: https://manuals.plus/asin/B07CPBCZHB.pdf
- TUL PCB envelope: https://www.tulembedded.com/fpga/images/PYNQ-Z2_PA_v2_pp_20201209_STD.pdf
