#!/bin/bash
#SBATCH -J effective_area
#SBATCH -c 1 # Number of cores
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --mem 6000
#SBATCH -t 0-07:00:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/holyscratch01/arguelles_delgado_lab/Everyone/jlazar/icecube_effa/output/icecube_effa_%A_%a.out # Standard output
#SBATCH -e /n/holyscratch01/arguelles_delgado_lab/Everyone/jlazar/icecube_effa/error/icecube_effa_%A_%a.err # Standard error
#SBATCH --array 1-100

N=1000
OUTPUT_PREFIX=/n/holyscratch01/arguelles_delgado_lab/Everyone/jlazar/icecube_effa/data
source /n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/setup.sh
CMD="python /n/home12/jlazar/hebe/paper_plots/effective_area/icecube_effa.py -s "${SLURM_ARRAY_TASK_ID}"  -n ${N} --output_prefix ${OUTPUT_PREFIX}"
echo ${CMD}
$CMD
