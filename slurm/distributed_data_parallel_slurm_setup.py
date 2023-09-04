import os
import socket
import argparse

# This script should run for python 2 and 3 without any packages.
# It targets linux because all SLURM clusters I know run Linux.

# It will output everything as a bash command that can then be used within the sbatch script

parser = argparse.ArgumentParser(description="Setup Pytorch Distributed Data Parallel (DDP) on a cluster managed by SLURM.")
parser.add_argument("--master_addr", required=True, type=str, help="Corresponds to MASTER_ADDR")
parser.add_argument("--master_port", required=True, type=int, help="Corresponds to MASTER_PORT")
parser.add_argument("--suppres_runtime_error", action='store_true',
    help="Suppress runtime error related to CUDA_VISIBLE_DEVICES. Only use if you know what you are doing.")

if __name__ == "__main__":
    args = parser.parse_args()

    # --- Check CUDA_VISIBLE_DEVICES ---
    #devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
   # if len(devices) != 1 and not args.suppres_runtime_error:
    #    raise RuntimeError("CUDA_VISIBLE_DEVICES={} does not have exactly one entry.".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    
    # --- Master Address and Port ---
    output = "export MASTER_ADDR='{}' && export MASTER_PORT='{}'".format(args.master_addr, str(args.master_port))

    # --- Compute world size ---
    # We assume one process per GPU, thus the world size = num_nodes*tasks_per_node
    num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    world_size = num_nodes * tasks_per_node
    output += " && export WORLD_SIZE='{}'".format(world_size)
    output += " && export LOCAL_WORLD_SIZE='{}'".format(tasks_per_node)

    # --- Set world rank ---
    world_rank = os.environ['SLURM_PROCID']
    output += " && export RANK='{}'".format(world_rank, world_rank)

    # --- Set Local Rank ---
    if world_size > 1:
        local_rank = os.environ['SLURM_LOCALID']
        output += " && export LOCAL_RANK='{}'".format(local_rank, local_rank)

    print(output)
