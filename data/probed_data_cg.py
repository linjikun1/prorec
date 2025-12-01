import pickle
import gzip
from tqdm import tqdm
from datasets import Dataset, DatasetDict

with gzip.open("original/cg_data_codeart.pkl.gz", "rb") as f:
    data = pickle.load(f)

total = len(data)
test_data = data[int(total * 0.9):]

test_src_list = []
test_codeart_list = []
test_callers_list = []
test_callees_list = []

for item in tqdm(test_data, total=len(test_data), desc="Processing test data"):
    test_src_list.append(item['source_code'])
    
    # 这里现在可以安全地工作了
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
    'src': test_src_list,
    'codeart': test_codeart_list, # 字典列表 (保留了所有键)
    'callers': test_callers_list, # 字典列表 (保留了所有键)
    'callees': test_callees_list  # 字典列表 (保留了所有键)
})

test_dataset = DatasetDict({'test': test_dataset})
test_dataset.save_to_disk('probed_data_cg/test')