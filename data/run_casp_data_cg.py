import pickle
import gzip
from tqdm import tqdm
from datasets import Dataset, DatasetDict

with gzip.open("/data3/linjk/study/work/prorec_aug/data/original/x64_O1/cg_data_codeart.pkl.gz", "rb") as f:
    data = pickle.load(f)

total = len(data)
train_data = data[:int(total * 0.8)]
valid_data = data[int(total * 0.8):int(total * 0.9)]

train_src_list = []
train_codeart_list = []
train_callers_list = []
train_callees_list = []

for item in tqdm(train_data, total=len(train_data), desc="Processing train data"):
    train_src_list.append(item['source_code'])
    
    train_codeart_list.append(repr({
        'code': item['codeart']['code'],
        'data_dep': item['codeart']['data_dep']
    }))
    
    callers_list = [repr({
        'code': caller['code'],
        'data_dep': caller['data_dep']
    }) for caller in item.get('callers', {}).values()]
    train_callers_list.append(callers_list)
    
    callees_list = [repr({
        'code': callee['code'],
        'data_dep': callee['data_dep']
    }) for callee in item.get('callees', {}).values()]
    train_callees_list.append(callees_list)

train_dataset = Dataset.from_dict({
    'src': train_src_list,
    'codeart': train_codeart_list, 
    'callers': train_callers_list, 
    'callees': train_callees_list  
})
train_dataset = DatasetDict({'train': train_dataset})
train_dataset.save_to_disk('/data3/linjk/study/work/prorec_aug/data/bimodal-lmpa-shuffled-cg/train')


valid_src_list = []
valid_codeart_list = []
valid_callers_list = []
valid_callees_list = []

for item in tqdm(valid_data, total=len(valid_data), desc="Processing valid data"):
    valid_src_list.append(repr(item['source_code']))
    
    valid_codeart_list.append(repr({
        'code': item['codeart']['code'],
        'data_dep': item['codeart']['data_dep']
    }))
    
    callers_list = [repr({
        'code': caller['code'],
        'data_dep': caller['data_dep']
    }) for caller in item.get('callers', {}).values()]
    valid_callers_list.append(callers_list)
    
    callees_list = [repr({
        'code': callee['code'],
        'data_dep': callee['data_dep']
    }) for callee in item.get('callees', {}).values()]
    valid_callees_list.append(callees_list)


valid_dataset = Dataset.from_dict({
    'src': valid_src_list,
    'codeart': valid_codeart_list,
    'callers': valid_callers_list,
    'callees': valid_callees_list
})
valid_dataset = DatasetDict({'valid': valid_dataset})
valid_dataset.save_to_disk('/data3/linjk/study/work/prorec_aug/data/bimodal-lmpa-shuffled-cg/valid')

print("数据划分完成")
