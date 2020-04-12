python -m torch.distributed.launch --nproc_per_node=4 train_epoch0.py > train_epoch0.txt
python -m torch.distributed.launch --nproc_per_node=4 train_epoch1.py > train_epoch1.txt
