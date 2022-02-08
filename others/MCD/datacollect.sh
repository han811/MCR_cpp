#!/bin/sh
while true
do 
    python datacollect.py --num $1 --total_iter $2
    sleep 3
done
