// pinkProcess desk enclosure R1. Units mm. F6 then export STL.
// part: shell, lid, stand, keeper, gauge, assembly
part="assembly";
$fn=40;
W=208; H=166; D=52; wall=2.8; face=2.8; lid_t=3.2;
window_w=158; window_h=92; window_y=0; pcb_t=1.6;
screen_pcb_back=8.4; // from exterior face; verify on sample
screen_pin_d=2.5; // fit pilot in the real display ears
function holes()=[[-78.45,53.48],[78.45,53.48],[-74.45,-53.48],[74.45,-53.48]];
module box(x,y,z,a,b,h){translate([x-a/2,y-b/2,z])cube([a,b,h]);}
module rr(w,h,r,z,t){translate([0,0,z])linear_extrude(t)offset(r=r)square([w-2*r,h-2*r],center=true);}
module peg(x,y,z,d,h){translate([x,y,z])cylinder(d=d,h=h);}
module xpin(x,y,z,d,h){translate([x,y,z])rotate([0,90,0])cylinder(d=d,h=h);}
module shell(){
 difference(){
  union(){
   difference(){
    rr(W,H,9,0,D);
    rr(W-2*wall,H-2*wall,6.2,face,D);
    translate([0,window_y,0])rr(window_w,window_h,2,-1,face+2);
    // Recessed lettering: front looks from -Z. Mirror X for readability.
    translate([0,-69,-0.01])mirror([1,0,0])linear_extrude(.61)
      text("pinkProcess",size=5.4,halign="center",valign="center",font="DejaVu Sans:style=Bold");
   }
   for(p=holes()){
    peg(p[0],p[1],face-.1,7.2,screen_pcb_back-pcb_t-face+.1);
    peg(p[0],p[1],screen_pcb_back-pcb_t-.1,screen_pin_d,6.8);
   }
  }
  for(s=[-1,1], x=[-68,68])box(x,s*(H/2-wall*.5),D-7.8,12,wall+2,3.6);
  // Open to back: plug externally then close; bundled cords exit left or right.
  for(s=[-1,1])box(s*W/2,0,D-24,wall*3,51,27);
  for(s=[-1,1],x=[-36:8:36])box(x,s*H/2,15,4,wall*3,16);
 }
}
// Retaining push collar. Side slit gives a compliant interference fit.
module keeper(){
 difference(){cylinder(d=6.8,h=2.5);peg(0,0,-1,screen_pin_d-.22,5);box(2.5,0,-1,5,.65,5);}
}
// All rear geometry local coordinates: +Z toward screen, -Z outside.
module latch(x,s){
 translate([x, s*(H/2-wall-.38),0])scale([1,s,1]){
  box(0,-.8,lid_t-.1,9.6,1.6,8.2);
  // ridge rises 1.25 into side capture window; slopes both sides
  translate([-4.8,0,0])rotate([90,0,90])linear_extrude(9.6)
   polygon([[-.25,lid_t+4.5],[1.25,lid_t+5.5],[1.25,lid_t+6.4],[-.25,lid_t+7.8]]);
 }
}
module board_saddle(x,s){
 // Board rests on short edge shelves; two long flexible fingers capture each side.
 translate([x,s*(87/2+.45),0])scale([1,s,1]){
  box(0,-1.4,lid_t-.1,7,5.8,5.6);
  box(0,.9,lid_t-.1,7,1.8,5.5+pcb_t+2);
  translate([-3.5,0,0])rotate([90,0,90])linear_extrude(7)
   polygon([[.2,lid_t+5.5+pcb_t+.25],[-1.05,lid_t+5.5+pcb_t+.25],[.2,lid_t+5.5+pcb_t+1.8]]);
 }
}
module lid(){
 difference(){
  union(){
   rr(W,H,9,0,lid_t);
   for(s=[-1,1],x=[-68,68])latch(x,s);
   for(s=[-1,1],x=[-59,59])board_saddle(x,s);
   for(s=[-1,1],y=[-38,38])box(s*71.25,y,lid_t-.1,1.8,7,7);
   for(s=[-1,1],x=[-85,85])box(x,s*(H/2-wall-1.4),lid_t-.1,10,2,3);
   // External pivot ears; bore axis is 5.4 mm behind outer face.
   for(x=[-51.7,51.7]){
    box(x,18,-5.4,4.8,11,5.6);
    xpin(x-2.4,18,-5.4,11,4.8);
   }
  }
  for(x=[-51.7,51.7])xpin(x-3,18,-5.4,4.8,6);
  for(y=[-58,56],x=[-33:6:33])box(x,y,-1,2.8,18,6);
 }
}
// U-shaped kickstand. The upper shoulders contact the lid to limit opening.
module stand(){
 difference(){
  union(){
   for(s=[-1,1]){
    box(s*45,18-(96-6)/2,-7.4,7.6,102,4);
    if(s==1)xpin(48.65,18,-5.4,4,3);
    if(s==-1)xpin(-51.65,18,-5.4,4,3);
   }
   box(0,-74,-7.4,97.6,8,4);
  }
 }
}
// One small print checks all four asymmetric hole locations.
module gauge(){
 difference(){
  union(){
   for(p=holes())peg(p[0],p[1],0,7.2,2);
   for(s=[-1,1])hull(){peg(s*78.45,53.48,0,3,2);peg(s*74.45,-53.48,0,3,2);}
   for(s=[-1,1])box(0,s*53.48,0,s==1?156.9:148.9,3,2);
  }
 }
 for(p=holes())peg(p[0],p[1],1.9,screen_pin_d,3.6);
}
if(part=="shell")shell();
if(part=="keeper")keeper();
if(part=="gauge")gauge();
if(part=="stand")translate([0,0,7.4])stand();
if(part=="lid")translate([0,0,H/2])rotate([90,0,0])lid();
if(part=="assembly"){
 color("lightpink")shell();
 translate([0,0,D+lid_t])rotate([0,180,0]){
 color("pink")lid();color("indianred")stand();}
}
