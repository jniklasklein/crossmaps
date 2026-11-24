# Simulation evaluation

The simulation launcher enables a separate ground-truth observer. It reads
Gazebo's model poses, not wheel odometry. Truth never enters ROS TF, the camera
adapter, perception, matching, pose guards or graph optimization. The dashboard
shows **Simulation evaluation** only for this simulation or a saved evaluation
report. Physical-rover runs have no truth panel.

## What to measure

| Question | Useful metric | Interpretation |
| --- | --- | --- |
| Does each rover's local pose drift? | Initial-pose-anchored translation/rotation RMSE and relative pose error (RPE) | Local baseline; independent origins are expected. |
| Are the two maps correctly placed? | Cross-rover relative pose error in metres/degrees | Report provisional and confirmed placements separately; unavailable is not zero error. |
| Does optimization improve accuracy? | Local-odometry baseline versus optimized poses, at the same timestamps with the same fixed reference | Not a PGO-disabled counterfactual experiment; both accuracy and internal consistency matter. |
| Are verified closures correct? | Ground-truth transform error and correctness fraction at explicit tolerances | Cross-rover measurements only; this is not detection recall. |
| Are reconstructed surfaces correct and complete? | Surface precision, completeness/recall and F-score at 5/10/20 cm | Requires independently sampled reference surfaces and a common observed-region mask; not yet implemented. |
| How much does the map overlap? | Occupied-voxel IoU at a fixed resolution and origin | Requires a common observed-space mask; unknown space must not count as empty. Not yet implemented. |
| Can rovers reconnect after exploring apart? | Time/distance to first correct alignment, drift since last rendezvous, false merges and recovery time | Requires repeatable moving trials; not established by a parked smoke test. |

Trajectory RMSE and RPE follow the distinction described in the
[TUM benchmark material](https://cvg.cit.tum.de/_media/spezial/bib/sturm12lecturenotes.pdf).
Our initial-pose-anchored error deliberately does **not** fit the whole
trajectory like standard fitted ATE: a continuously refitted alignment could
hide drift. There is no scale fitting. Surface accuracy/completeness/F-score
are described by [ETH3D](https://eth3d.ethz.ch/high_res_multi_view.php?metric=accuracy).
IoU alone is a poor headline metric for a sparse, segmented surface map.

## Scope and limitations of the first implementation

- Truth is associated at retained keyframe **simulation capture timestamps**,
  using bounded interpolation, never arrival time or extrapolation. Optical
  camera truth uses the generated model's declared mount transform.
- Local results are per source session. Shared estimates use a fixed common
  reference; independently fitting each rover would conceal a wrong merge.
- RPE uses consecutive retained keyframes and reports their actual time-interval
  range. It is **not** a fixed-one-second or full-rate odometry benchmark.
- Cross-rover relative errors compare retained capture pairs, which may have
  different capture times. They are not instantaneous live robot separation.
- Optimization comparison is a raw-local-pose baseline versus current optimized
  estimates, not a claim about the change from one individual solver iteration.
- CrossMaps currently verifies cross-rover closures. Single-rover loop closures
  occur in SLAM Toolbox and are not evaluated by this panel.
- Missing/stale truth, insufficient samples and unaligned maps are shown as
  unavailable. Always inspect association coverage and sample counts.
- This measures consistency with simulated geometry. Sensor noise, contact,
  wheel slip and inertias are not calibrated to the physical rover, so simulator
  accuracy is not a real-world accuracy guarantee.

## Reproducible experiments to add next

Use the same world, spawn poses, camera settings, sensor profile, random seed
and route in each comparison. Keep independent truth outside the pipeline.

1. **Calibration sanity:** parked rovers, then measured straight motion and a
   turn. Confirm transform conventions, units and capture-time association.
2. **Shared-view initialization:** partial overlap, different viewpoints,
   repeated textures and an intentionally non-overlapping negative case.
3. **Separate exploration:** one initial rendezvous, diverging routes, then a
   return encounter. Plot error versus travelled distance and time since contact.
4. **Sparse rendezvous:** larger multi-room routes with zero, one and several
   encounters. Include failed alignment trials in coverage/results rather than
   evaluating only successful merges.
5. **Closure stress:** revisit with changed heading, occlusion and sensor noise.
   Score false accepted constraints and correction magnitude as well as RMSE.
6. **Paired ablations:** repeat identical seeded routes with PGO disabled/enabled,
   provisional placement disabled/enabled and differing rendezvous frequency.
   Compare identical associated samples; do not compare unlike routes.

True closure recall needs an independent visibility/overlap oracle defining
which pairs *could* match. Full-map surface metrics need a reference sampler
and visibility mask. Those are deliberate next steps, not inferred from a high
retrieval score, low graph residual or a visually pleasing overlay.

## Storage and replay

The truth sidecar overwrites a bounded rolling buffer (300 samples per rover).
The coordinator retains truth associations only for bounded map evidence and
overwrites one latest `evidence/evaluation.json` summary. No unbounded pose log
is created. The summary is included in recording storage accounting, displayed
as recorded rather than live on replay, and cleared with **Rebuild current map**.
Older reports are not imported into a new run.
