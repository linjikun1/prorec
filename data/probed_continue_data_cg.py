import pickle
import gzip
import json
from tqdm import tqdm
from datasets import Dataset, DatasetDict

with gzip.open("/data3/linjk/study/work/prorec_aug/data/original/x64_O1/cg_data_codeart.pkl.gz", "rb") as f:
    data = pickle.load(f)

total = len(data)
test_data = data[int(total * 0.9):]

with open('/data3/linjk/study/work/prorec_aug/save/scored_signatures.json', 'r') as f:
    scored_signatures = json.load(f)

test_info_list = []
test_codeart_list = []
test_callers_list = []
test_callees_list = []

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
    
    callers_list = [repr({
        'code': caller['code'],
        'data_dep': caller['data_dep']
    }) for caller in item.get('callers', {}).values()]
    test_callers_list.append(callers_list)
    
    callees_list = [repr({
        'code': callee['code'],
        'data_dep': callee['data_dep']
    }) for callee in item.get('callees', {}).values()]
    test_callees_list.append(callees_list)

test_dataset = Dataset.from_dict({
    'codeinfo': test_info_list,
    'codeart': test_codeart_list,
    'candidate_signatures': scored_signatures,
    'callers': test_callers_list, # 字典列表 (保留了所有键)
    'callees': test_callees_list  # 字典列表 (保留了所有键)
})

test_dataset = DatasetDict({'test': test_dataset})
test_dataset.save_to_disk('/data3/linjk/study/work/prorec_aug/data/probed_continue_data_cg/test')