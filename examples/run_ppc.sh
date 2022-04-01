#!/bin/bash
for (( counter=1; counter>0; counter-- ))
do
python example_ppc.py $counter
done
printf "\n"