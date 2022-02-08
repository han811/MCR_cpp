#!/bin/sh
is_given=1
while true
do 
    python test.py --given $is_given
    sleep 3
done
