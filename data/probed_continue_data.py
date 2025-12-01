from datasets import Dataset, DatasetDict
from tqdm import tqdm
import json

with open('original/test.json', 'r') as f:
    test_data = json.load(f)
# with open('/home/linjk/study/work/prorec.json', 'r') as f:
#     test_data = json.load(f)
with open('../save/scored_signatures.json', 'r') as f:
    scored_signatures = json.load(f)

test_info_list = []
test_codeart_list = []

for item in tqdm(test_data, total=len(test_data), desc="Processing test data"):
    test_info_list.append(repr({
        # 'projec_name': item['codeart']['metadata']['project_name'],
        # 'function_name': item['codeart']['metadata']['function_name'],
        # 'func_addr': item['codeart']['metadata']['func_addr'],
        'decompiled_code': item['codeart']['strip_decompiled_code'],
        'comment': item['comment']
    }))
    test_codeart_list.append(repr({
        'code': item['codeart']['code'],
        'data_dep': item['codeart']['data_dep']
    }))

test_dataset = Dataset.from_dict({
    'codeinfo': test_info_list,
    'codeart': test_codeart_list,
    'candidate_signatures': scored_signatures
})
test_dataset = DatasetDict({'test': test_dataset})
test_dataset.save_to_disk('probed_continue_data/test')