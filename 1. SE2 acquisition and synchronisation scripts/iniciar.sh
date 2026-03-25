#!/bin/bash
sleep 15

# Activate virtualenv
source /home/pi/Emberometer/env/env/bin/activate

# First: wait for GPS fix, logging attempts every ~30s
python3 /home/pi/Emberometer/gps_wait.py
GPS_STATUS=$?

# Now start your main Emberometer run
python3 /home/pi/Emberometer/run.py

