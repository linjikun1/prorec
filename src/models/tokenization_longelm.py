from copy import deepcopy
import networkx as nx
from networkx.algorithms.shortest_paths import floyd_warshall_numpy
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

import time


class LongelmTokenizer(PreTrainedTokenizerFast):

    def __init__(
        self,
        node_size=1,
        block_size=8,
        max_blocks=200,
        global_memory_size=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.node_size = node_size
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.global_memory_size = global_memory_size

        self.inst_token = '<INST>'

    def create_relative_node_positions(
        self,
        block_count,
        deps
    ):
        # filter out out-of-range dependencies
        assert block_count <= self.max_blocks
        remaining_data_dep = []
        for source_node_id, target_node_id in deps:
            if source_node_id >= block_count or target_node_id >= block_count:  # support truncation
                continue
            else:
                remaining_data_dep.append((source_node_id, target_node_id))

        # graph construction
        graph = nx.Graph()
        graph.add_nodes_from(range(block_count))
        graph.add_edges_from(remaining_data_dep)

        # dense graph all pairs shortest path length
        spl_matrix = floyd_warshall_numpy(graph)
        spl_matrix[spl_matrix == np.inf] = -1
        spl_matrix = torch.tensor(
            spl_matrix, dtype=torch.long)

        # TODO: expand as node size and reshape as final matrix

        return spl_matrix

    def inst_encode(
        self,
        code: List[Tuple[int, str]],
        deps: List[Tuple[int, int]],
        return_extra_info=False,
    ):
        tokens = []
        block_count = 0
        special_tokens_mask = []
        
        for inst_id, instruction in code:   # NOTE: this can be accelerated by batch encode
            if inst_id >= self.max_blocks:
                break
            instruction_tokens = self.tokenize(instruction)
            block_count += 1

            # pad or truncate block
            instruction_tokens = instruction_tokens[:self.block_size]
            tokens += instruction_tokens
            special_tokens_mask += [0] * len(instruction_tokens)
            if len(instruction_tokens) < self.block_size:
                tokens += [self.pad_token] * (self.block_size - len(instruction_tokens))
                special_tokens_mask += [1] * (self.block_size - len(instruction_tokens))

        # pad blocks to max_blocks
        # (NOTE: in practice, just make sure each instance in batch has same block_count)
        if block_count <= self.max_blocks:
            tokens += [self.pad_token] * self.block_size * (self.max_blocks - block_count)
            special_tokens_mask += [1] * self.block_size * (self.max_blocks - block_count)

        attention_mask = (~torch.tensor(special_tokens_mask, dtype=torch.bool)).int()

        # nodes
        tokens += [self.inst_token] * self.max_blocks * self.node_size
        special_tokens_mask += [1] * self.max_blocks * self.node_size
        
        # global memory
        tokens += [self.cls_token] * self.global_memory_size
        special_tokens_mask += [1] * self.global_memory_size

        # convert tokens to ids
        input_ids = self.convert_tokens_to_ids(tokens)

        relative_node_positions = self.create_relative_node_positions(
            self.max_blocks,
            deps
        )

        if return_extra_info:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long), \
                'attention_mask': attention_mask, \
                'special_tokens_mask': torch.tensor(special_tokens_mask, dtype=torch.int), \
                'relative_node_positions': relative_node_positions
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long), \
                'attention_mask': attention_mask, \
                'relative_node_positions': relative_node_positions
            }
    
    def batch_inst_encode(
        self,
        examples,
        max_transitions=None,
    ):
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'graph_attention_mask': [],
            'relative_node_positions': []
        }
        
        for example in examples:
            if isinstance(example['code'], str):
                encoded = self.inst_encode(eval(example['code']), eval(example['data_dep']))
            else:
                encoded = self.inst_encode(example['code'], example['data_dep'])
            batch['input_ids'].append(encoded['input_ids'])
            batch['attention_mask'].append(encoded['attention_mask'])
            batch['graph_attention_mask'].append(encoded['relative_node_positions'] >= 0)
            batch['relative_node_positions'].append(encoded['relative_node_positions'])
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'graph_attention_mask': torch.stack(batch['graph_attention_mask']),
            'relative_node_positions': torch.stack(batch['relative_node_positions'])
        }
    
    def batch_inst_encode_cg(
        self,
        examples,
        max_transitions=None,
    ):
        import ast
        def process(item):
            # if isinstance(item['code'], str):
            #     encoded = self.inst_encode(eval(item['code']), eval(item['data_dep']))
            # else:
            while isinstance(item, str):
                item = ast.literal_eval(item)
            encoded = self.inst_encode(item['code'], item['data_dep'])
            return encoded

        # --- GNN 批处理所需的新列表 ---
        # 1. Longelm (CodeArt) 输入
        longelm_inputs_list = []  # 存储所有节点 (target+callers+callees) 的 'encoded' 字典
        
        # 2. GNN 图结构输入
        edge_index_list = []       # 存储 (src, dst) 边元组
        target_node_indices = []   # 存储每个样本的 target 节点在批次中的全局索引
        batch_indices = []         # 存储每个节点属于哪个原始样本
        
        node_counter = 0           # 跟踪批次中的全局节点索引

        # --- 遍历批次中的每个样本 (example) ---
        for i, example in enumerate(examples):
            # 1. 收集当前图的所有节点 (确保数据存在)
            target_node = example.get('codeart') 
            # 如果 'codeart' 不存在或为空，跳过这个样本
            if not target_node:
                continue

            caller_nodes = example.get('callers', [])[:5]
            callee_nodes = example.get('callees', [])[:5]
            
            nodes_in_current_graph = [target_node] + caller_nodes + callee_nodes
            num_nodes = len(nodes_in_current_graph)

            # 2. 确定 target 节点的全局索引
            # 假设 target 始终是每个图的第一个节点
            target_global_idx = node_counter
            target_node_indices.append(target_global_idx)

            # 3. 为 GNN 创建图结构 (边)
            # 边: Callers -> Target
            for j in range(len(caller_nodes)):
                caller_local_idx = j + 1
                caller_global_idx = node_counter + caller_local_idx
                edge_index_list.append((caller_global_idx, target_global_idx))
            
            # 边: Target -> Callees
            for j in range(len(callee_nodes)):
                callee_local_idx = j + 1 + len(caller_nodes)
                callee_global_idx = node_counter + callee_local_idx
                edge_index_list.append((target_global_idx, callee_global_idx))

            # 4. 处理所有节点，为 Longelm 准备输入
            for node in nodes_in_current_graph:
                longelm_inputs_list.append(process(node))

            # 5. 为 GNN 创建 batch 索引
            batch_indices.append(torch.full((num_nodes,), i, dtype=torch.long))
            
            # 6. 更新全局节点计数器
            node_counter += num_nodes

        # --- 批处理结束，开始将所有数据堆叠成张量 --- 


        # 1. 堆叠 Longelm/CodeArt 的输入
        final_input_ids = torch.stack([d['input_ids'] for d in longelm_inputs_list])
        final_attention_mask = torch.stack([d['attention_mask'] for d in longelm_inputs_list])
        final_rel_pos = torch.stack([d['relative_node_positions'] for d in longelm_inputs_list])
        final_graph_attn_mask = (final_rel_pos >= 0)

        # 2. 堆叠 GNN 的输入
        if edge_index_list:
            final_edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            final_edge_index = torch.empty((2, 0), dtype=torch.long)
            
        final_batch_indices = torch.cat(batch_indices)
        final_target_node_indices = torch.tensor(target_node_indices, dtype=torch.long)

        # 3. 返回一个全新的字典，包含 GNN 所需的所有信息
        return {
            'longelm_input_ids': final_input_ids,
            'longelm_attention_mask': final_attention_mask,
            'longelm_graph_attention_mask': final_graph_attn_mask,
            'longelm_relative_node_positions': final_rel_pos,
            
            'gnn_edge_index': final_edge_index,
            'gnn_batch_index': final_batch_indices,
            'gnn_target_node_indices': final_target_node_indices,
            
            'return_loss': True # 保持不变，用于模型
        }
