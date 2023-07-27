#!/bin/bash

# number of runs to perform
NUM_RUNS=10

# loop to run python script
for ((i=1; i<=NUM_RUNS; i++))
do
    echo "Run $i:"
    /home/pradeep/miniconda3/envs/tf/bin/python d_net_dqn_diff_goal_obst_eleven_modified.py
    sleep 3
done