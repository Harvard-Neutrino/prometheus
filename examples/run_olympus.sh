#!/bin/bash
for (( counter=996; counter>0; counter-- ))
do
python example_olympus.py $counter
done
printf "\n"