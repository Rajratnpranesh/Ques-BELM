python -m torch.distributed.launch --nproc_per_node=4 train_epoch0.py > train_epoch0.txt
python valid_epoch0.py
python nq_eval.py --gold_path=../../input/natural_questions/v1.0-simplified_nq-dev-all.jsonl.gz --predictions_path=predictions_epoch0.json > eval_epoch0.txt
python -m torch.distributed.launch --nproc_per_node=4 train_epoch1.py > train_epoch1.txt
python valid_epoch1.py
python nq_eval.py --gold_path=../../input/natural_questions/v1.0-simplified_nq-dev-all.jsonl.gz --predictions_path=predictions_epoch1.json > eval_epoch1.txt
python -m torch.distributed.launch --nproc_per_node=4 train_epoch2.py > train_epoch2.txt
python valid_epoch2.py
python nq_eval.py --gold_path=../../input/natural_questions/v1.0-simplified_nq-dev-all.jsonl.gz --predictions_path=predictions_epoch2.json > eval_epoch2.txt
