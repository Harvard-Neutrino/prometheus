#!/bin/bash
for (( counter=1000; counter>0; counter-- ))
do
python example_olympus.py $counter
done
printf "\n"