"use strict";
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { anchorTransform: tf, buildRoverOverlay: overlay, projectMarker, drawRoverOverlay, roverPoseLegend } = require("../collab_dashboard/app.js");
const I = tf(0,0,0,0);
const robot = { robot_id:"A", session_id:"s", aligned:true, map_observation_age_s:1, trajectory:{
  latest:{submap_id:"m", seq:2, frame:"base", T_submap_pose:tf(2,0,0,0)},
  segments:[{submap_id:"m", break_before:true, frame:"base", points:[[0,0,0],[2,0,0]]}]} };
const sm = {robot_id:"A", session_id:"s", id:"m", aligned:true, T_local_submap:tf(1,0,0,0), T_world_submap:tf(10,5,0,90)};
function state(r=robot, maps=[sm]) { return {robots:[r],submaps:maps, shadow_mode:false}; }
function approx(a,b){assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-8,`${i}: ${v} ≠ ${b[i]}`));}
test("marker and path follow world re-optimization or selected local origin; base +X forward",()=>{
  const world=overlay(state()), local=overlay(state(),"A");
  approx(world.markers[0].position,[10,7,0]);approx(world.markers[0].forward,[0,1,0]);approx(world.trails[0].points[0],[10,5,0]);
  approx(local.markers[0].position,[3,0,0]);approx(local.markers[0].forward,[1,0,0]);
  approx(overlay(state(robot,[{...sm,T_world_submap:tf(-4,-2,0,0)}])).markers[0].position,[-2,-2,0]);
});
test("provisional requires a separate valid preview transform, automatic mode, and visible preview",()=>{
 const m={...sm,aligned:false,alignment_state:"provisional",T_preview_world_submap:tf(3,0,0,0)}, s=state(robot,[m]);
 const out=overlay(s);approx(out.markers[0].position,[5,0,0]);assert.equal(out.markers[0].tentative,true);assert.equal(out.trails[0].tentative,true);
 for (const [snap,opts] of [[s,{showProvisional:false}],[{...s,shadow_mode:true},{}],[state(robot,[{...m,T_preview_world_submap:null}]),{}],[state(robot,[{...m,alignment_state:"unaligned"}]),{}]]) {
   assert.deepEqual(overlay(snap,"combined",opts),{markers:[],trails:[]});
 }
 assert.equal(overlay(s,"A").markers[0].tentative,false);
});
test("never connect through session reset, omitted submap, refused tracking, or a frame-type switch",()=>{
 const seg=(id,start,br=false,frame="base")=>({submap_id:id,break_before:br,frame,points:[[start,0,0],[start+1,0,0]]});
 const s=state({...robot,trajectory:{...robot.trajectory,segments:[seg("m",0),seg("k",1),seg("old",2),seg("k",3),seg("k",4,true),seg("k",5,false,"camera_optical")]}},[sm,{...sm,id:"k"},{...sm,id:"old",session_id:"past"}]);
 assert.deepEqual(overlay(s,"A").trails.map(run=>run.points.length),[4,2,2,2]);
 const unknown=state({...robot,session_id:"new"});assert.deepEqual(overlay(unknown),{markers:[],trails:[]});
});
test("last-known and recorded markers remain honest, controls are independent, camera uses optical +Z",()=>{
 const r={...robot,trajectory:{...robot.trajectory,latest:{...robot.trajectory.latest,frame:"camera_optical"}}};
 let m=overlay(state(r)).markers[0];approx(m.forward,[0,0,1]);assert.match(roverPoseLegend(r,m,"combined",true),/camera pose \(base unavailable\)/);
 for(const rated of [{...r,map_observation_age_s:10},{...r,trajectory:{...r.trajectory,tracking_lost:true}},{...r,pose_quality:{integration_allowed:false}}])
   assert.equal(overlay(state(rated)).markers[0].stale,true);
 m=overlay({...state({...r,map_observation_age_s:100}),mode:"recorded_replay"}).markers[0];
 assert.equal(m.stale,false);assert.match(roverPoseLegend(r,m,"A",true),/recorded camera/);
 assert.equal(overlay(state(),"combined",{connected:false}).markers[0].stale,true);
 let out=overlay(state(),"combined",{showMarkers:false});assert.equal(out.markers.length,0);assert.equal(out.trails.length,1);
 out=overlay(state(),"combined",{showTrails:false});assert.equal(out.markers.length,1);assert.equal(out.trails.length,0);
 assert.deepEqual(overlay(state({...robot,trajectory:null})),{markers:[],trails:[]});
});
test("2D projection matches WebGL clip space; controls are visible and overlay doesn't absorb map dragging",()=>{
 const fs=require("node:fs"),path=require("node:path"),dir=path.resolve(__dirname,"../collab_dashboard");
 const html=fs.readFileSync(path.join(dir,"index.html"),"utf8"),css=fs.readFileSync(path.join(dir,"styles.css"),"utf8");
 approx(projectMarker([0,0,0],I.flat(),80,60),[40,30]);approx(projectMarker([.5,.5,0],I.flat(),80,60),[60,15]);
 assert.equal(projectMarker([0,0,-2],I.flat(),80,60),null);
 for (const id of ["show-rovers","show-trails","map-poses"]) assert.ok(html.includes(`id="${id}"`));
 assert.match(css,/#map-poses\{[^}]*pointer-events:none/);
 const calls=[],ctx=new Proxy({measureText(){return {width:8};}}, {get:(o,k)=>o[k]||((...args)=>calls.push([k,...args]))});
 const testOverlay={markers:[{...overlay(state()).markers[0],position:[0,0,0],forward:[1,0,0]}], trails:[{tentative:true,points:[[0,0,0],[.1,0,0]]}]};
 drawRoverOverlay({},ctx,testOverlay,I.flat(),80,60,1);assert.ok(calls.some(c=>c[0]==="fillText"&&c[1]==="A"));assert.ok(calls.some(c=>c[0]==="setLineDash"&&c[1].length===2));
});
