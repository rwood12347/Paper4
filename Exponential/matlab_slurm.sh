#!/bin/bash -l
#SBATCH --time=00:5:00
#SBATCH --mem-per-cpu=3G
#SBATCH -o ExponentialSparseTimes.out
module load matlab
experiments=1
min_nodes=10
max_nodes=100
node_skip=10
time_frames=2
seed=42
isSparse=1
echo "Experiments:$experiments"
srun matlab -nojvm -nosplash -batch "experiment_loop($experiments, $min_nodes, $max_nodes, $node_skip, $time_frames, $s>

