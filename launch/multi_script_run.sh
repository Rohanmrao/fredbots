#! /usr/bin/bash

# this runs the q learning scripts in parallel

python3 /home/rohan/catkin_ws/src/fredbots/src/scripts/ql_bot_1.py &
python3 /home/rohan/catkin_ws/src/fredbots/src/scripts/ql_bot_2.py &
python3 /home/rohan/catkin_ws/src/fredbots/src/scripts/ql_bot_3.py &
python3 /home/rohan/catkin_ws/src/fredbots/src/scripts/ql_bot_4.py &

