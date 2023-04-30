#!/bin/bash
BAG_DIR=$MOBILE_MANIPULATION_CENTRAL_BAG_DIR/$(date +"%Y-%m-%d")
mkdir -p "$BAG_DIR"

rosbag record -o "$BAG_DIR/$1" \
  /clock \
  --regex "/ridgeback/(.*)" \
  --regex "/ridgeback_velocity_controller/(.*)" \
  --regex "/vicon/(.*)" \
  --regex "/wrench/(.*)"
