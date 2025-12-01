from datasets import Dataset, DatasetDict
from tqdm import tqdm
import json

with open('original/train.json', 'r') as f:
    train_data = json.load(f)

with open('original/valid.json', 'r') as f:
    valid_data = json.load(f)

train_src_list = []
train_codeart_list = []

for item in tqdm(train_data, total=len(train_data), desc="Processing train data"):
    train_src_list.append(repr(item['src']))
    train_codeart_list.append(repr({
        'code': item['codeart']['code'],
        'data_dep': item['codeart']['data_dep']
    }))

train_dataset = Dataset.from_dict({
    'src': train_src_list,
    'codeart': train_codeart_list
})
train_dataset = DatasetDict({'train': train_dataset})
train_dataset.save_to_disk('bimodal-lmpa-shuffled/train')


valid_src_list = []
valid_codeart_list = []

for item in tqdm(valid_data, total=len(valid_data), desc="Processing valid data"):
    valid_src_list.append(repr(item['src']))
    valid_codeart_list.append(repr({
        'code': item['codeart']['code'],
        'data_dep': item['codeart']['data_dep']
    }))

valid_dataset = Dataset.from_dict({
    'src': valid_src_list,
    'codeart': valid_codeart_list
})
valid_dataset = DatasetDict({'valid': valid_dataset})
valid_dataset.save_to_disk('bimodal-lmpa-shuffled/valid')

print("数据划分完成")