#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition= #your_partition
#SBATCH -J job_name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2

#setup your environment here

######

# --- Setup In Master ---
# Master will be the one running this bash script (SLURM runs this only once)
# Get hostname and port on first node first process
# For the port see: https://unix.stackexchange.com/questions/55913/whats-the-easiest-way-to-find-an-unused-local-port
master_addr=$(hostname -i)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')


# --- Call the Script which the User will Edit ---
# With srun this will be run on all nodes for all processes
srun ./distributed_data_parallel_slurm_run.bash --master_addr $master_addr --master_port $master_port
