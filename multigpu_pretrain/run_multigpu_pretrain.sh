#!/bin/sh

dist='torch.distributed.launch'
python_path='../'
python='python'
executable='./multigpu_pretrain.py'

# distributed arguments
devices='0,1'
nproc_per_node=2
master_port=1234

CUDA_VISIBLE_DEVICES=$devices PYTHONPATH=$PYTHONPATH:$python_path \
        $python -m $dist --nproc_per_node=$nproc_per_node --master_port=$master_port $executable \
        --seed=0
