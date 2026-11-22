"""Desk prototype, mm. Adjust trial mount positions against physical hardware."""
import cadquery as cq
from pathlib import Path
out=Path(__file__).parent
def block(x,y,z,w,h,d):
    return cq.Workplane("XY").box(w,h,d,centered=(False,False,False)).translate((x,y,z))
def shellbox(x,y,z,w,h,d,r):
    return block(x,y,z,w,h,d).edges('|Z').fillet(r)
def yz(x,w,pts):
    return cq.Workplane('YZ').polyline(pts).close().extrude(w).translate((x,0,0))
W,H=172,134
left,bottom=19,8
def bodybox(x,y,z,w,h,d,r):
    return shellbox(left+x,bottom+y,z,w,h,d,r)
front=bodybox(0,0,0,W,H,29,7).cut(bodybox(2.4,2.4,2.4,W-4.8,H-4.8,28,4.6))
front=front.cut(shellbox(27,29,-1,156,92,5,2))
# Four local ear hooks. Trial 1.8 mm seat gap, 1.0 mm engagement.
for x in (27,177):
    for lower in (True,False):
        edge=12.865 if lower else 150-12.865
        sign=1 if lower else -1
        def y(d):return edge+sign*d
        pts=[(y(-2),2.35),(y(2.8),2.35),(y(2.8),7),(y(0),7),(y(0),8.8),(y(1),8.8),(y(1),9.3),(y(-.3),11.2),(y(-2),11.2)]
        front=front.union(yz(x,6,pts))
rear=bodybox(0,0,31,W,H,11,7).cut(bodybox(2.4,2.4,30,W-4.8,H-4.8,9.6,4.6))
# End-supported raised tracks. Board slides from right before closing.
for base,sgn in [(31,1),(119,-1)]:
    def y(d):return base+sgn*d
    profile=[(y(-2),39.7),(y(2),39.7),(y(2),35),(y(0),35),(y(0),32.8),(y(1.4),32.8),(y(1.4),31.4),(y(-2),31.4)]
    for xx in (35,88,147):rear=rear.union(yz(xx,14,profile))
rear=rear.union(block(32,46,31.5,2.6,12,8.2))
# Board retention tab at entry end with chamfered lead-in; manually deflect to insert.
rear=rear.union(block(175,57,32,1.5,8,7.7)).union(block(174.3,57,32,1,8,1))
for x in (36,158):
    for y,sgn in [(bottom+2.4,-1),(bottom+H-2.4,1)]:
        prof=[(y-sgn*1.7,32),(y-sgn*1.7,23),(y-sgn*.5,23),(y+sgn*.9,25.1),(y+sgn*.9,26),(y-sgn*.2,26),(y-sgn*.2,32)]
        # connect latch roots to rear roof
        rear=rear.union(yz(x,12,prof)).union(block(x,min(y-sgn*1.7,y-sgn*.2),31.5,12,1.5,8.2))
        front=front.cut(block(x-.4,bottom-1 if sgn<0 else bottom+H-3,20.7,12.8,4,3.7))
# Side seam cable gates; cables lay into them before closure.
for x in (left-1,left+W-4):
    front=front.cut(block(x,42,9,5,56,23))
    rear=rear.cut(block(x,42,30,5,56,8))
for x in range(60,150,9):
    for y in (bottom-1,bottom+H-3):
        front=front.cut(block(x,y,16,4,4,10))
# Flat rear tray keeps the desk profile to 40 mm.
# SD card is beneath the board; expose its edge directly through the back.
# Board is components toward panel, Ethernet toward X=0, SD toward +Y.
rear=rear.cut(shellbox(115,110,33,28,27,12,4))
# Carry the rose surround over the tray seam; retain 0.4 mm running clearance.
skirt=bodybox(-2,-2,27.5,W+4,H+4,10.8,9).cut(bodybox(-.4,-.4,27,W+.8,H+.8,13,7.4))
shoulder=bodybox(-2,-2,26,W+4,H+4,2.2,9).cut(bodybox(2.4,2.4,25,W-4.8,H-4.8,5,4.6))
front=front.union(shoulder).union(skirt)
for x in (left-4,left+W-5):front=front.cut(block(x,42,9,10,56,35))
# Wordmark engraved into the face. Text reads with the screen in landscape.
mark=cq.Workplane("XY").text("pinkProcess",6,.55,font="DejaVu Sans",combine=True).rotate((0,0,0),(1,0,0),180).translate((105,130,.45))
front=front.cut(mark)
rear=rear.translate((0,0,-2))
# Passive side vents at the cooler end of the tray.
for y in range(103,127,5):
    vent=block(left+W-5,y,13,9,2.8,13)
    front=front.cut(vent)
# Cropped actual features provided as separate small fit samples.
ear=front.intersect(block(21,7,0,18,14,13))
rail=rear.intersect(block(31,25,28,23,12,12))
parts={'Front_bezel':front,'Rear_tray':rear,'Test_LCD_ear':ear,'Test_PCB_rail':rail}
for name,p in parts.items():
    assert p.val().isValid() and p.solids().size()==1,name
    cq.exporters.export(p,str(out/(name+'.stl')))
assert front.val().intersect(rear.val()).Volume()<.001
asm=cq.Assembly();asm.add(front,name='Front');asm.add(rear,name='Rear');asm.save(str(out/'Desk_console.step'))
