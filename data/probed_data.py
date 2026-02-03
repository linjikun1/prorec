from datasets import Dataset, DatasetDict
from tqdm import tqdm
import json

with open('original/test.json', 'r') as f:
    test_data = json.load(f)
# with open('/home/linjk/study/work/prorec.json', 'r') as f:
#     test_data = json.load(f)

test_dec_list = []
test_codeart_list = []

for item in tqdm(test_data, total=len(test_data), desc="Processing test data"):
    test_dec_list.append(repr(item['codeart']['strip_decompiled_code']))
    test_codeart_list.append(repr({
        'code': item['codeart']['code'],
        'data_dep': item['codeart']['data_dep']
    }))

test_dataset = Dataset.from_dict({
    'decompiled_code': test_dec_list,
    'codeart': test_codeart_list
})
test_dataset = DatasetDict({'test': test_dataset})
test_dataset.save_to_disk('probed_data/test')