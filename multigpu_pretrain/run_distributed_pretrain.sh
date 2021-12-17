#!/bin/sh

dist='torch.distributed.launch'
python_path='../'
python='python'
executable='./multigpu_pretrain.py'

# Local machine settings.
devices='0,1'  # Devices to use on local machine.
nproc_per_node=2  # Number of worker processes on local machine. Should be the same on all nodes.

# Cluster settings.
nnodes=2  # Total number of compute nodes to use.
node_rank=0  # Rank of the current node (node index starting from 0, like 0, 1, 2, ...).

# Master node settings (node with rank 0).
master_addr="192.168.0.1"  # IP address of master node. Use 127.0.0.1 (loopback) for single-node training.
master_port=7777  # Master node open port to use for multi-node communication.

# Launching pre-training script in multi-gpu setting.
#
# The following environment variables are set manually:
# 1) Setting CUDA_VISIBLE_DEVICES variable all PyTorch to use only specific GPU's.
# 2) Added path to local sources in PYTHONPATH variable.
# 3) Specific NCCL environment variables. You should use your own NCCL settings, these just worked for us.
#    Read more about them here: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
CUDA_VISIBLE_DEVICES=$devices \
    NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0 \
    PYTHONPATH=$PYTHONPATH:$python_path \
    $python -m $dist \
        --nproc_per_node=$nproc_per_node \
        --nnodes=$nnodes \
        --node_rank=$node_rank \
        --master_addr=$master_addr \
        --master_port=$master_port \
        $executable \
            --seed=0
