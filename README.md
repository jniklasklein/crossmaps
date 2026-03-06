# CrossMaps
## A Confidence-Aware Open-Vocabulary Semantic Mapping for Rover Navigation

For Jetson-based UGV Waveshare rovers

To load the presaved semantic map (in /data): Put the *crossmaps_3_node_state.pkl* file into a directory ~/.crossmaps


These are the parameters I use to run the system, you can use them to load the saved semantic map too:
(**Important** - the time decay will cause the STM to appear empty if you load it with these parameters. You need to either set ```decay_half_life_s``` several orders of magnitude higher (eg. 1200000) or comment out the time decay before running (line 770: ```self.stm_w[key] *= decay```). The LTM should be stable either way.)

<pre>
python ~/vlmaps_ws/crossmaps_6_node.py --ros-args \
  -p internal_frame:=odom \
  -p publish_frame:=map \
  -p use_ray_consistency:=true \
  -p occ_grid_topic:=/map \
  -p ray_check_stride:=6 \
  -p resolution:=0.10 \
  -p embed_every_n:=8 \
  -p stride:=8 \
  -p gating_cos_min:=0.20 \
  -p coherence_floor:=0.20 \
  -p coherence_power:=2.0 \
  -p use_visibility_boost:=true \
  -p view_bins:=8 \
  -p view_boost_strength:=0.7 \
  -p decay_half_life_s:=120.0 \
  -p cell_weight_max:=50.0 \
  -p ltm_weight_max:=200.0 \
  -p ltm_update_gain:=1.2 \
  -p ltm_replace_cos_thresh:=0.25 \
  -p ltm_replace_gain:=3.0 \
  -p promote_min_conf:=0.55 \
  -p promote_min_coh01:=0.40 \
  -p promote_min_views:=2 \
  -p autoload_map_state:=true
</pre>


  Then, in a second terminal and open ```rviz2```

  You can set queries in a third terminal with:
<pre>
 ros2 param set /crossmaps_3_node query_text plant 
</pre>

  
