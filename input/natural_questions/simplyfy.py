import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import jsonlines
from test_utils import simplify_nq_example
import json


json_dir = 'v1.0-simplified_nq-dev-all.jsonl'
dict_list = []
with open(json_dir) as f:
    for line in tqdm(f):
        dict_list.append(simplify_nq_example(json.loads(line)))

with jsonlines.open('simplified-nq-valid.jsonl', 'w') as writer:
    writer.write_all(dict_list)
