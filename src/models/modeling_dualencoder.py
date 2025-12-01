# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch (Assembly-Source Code) DualEncoder model."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
import torch.distributed as dist

# --- 1. 添加 GNN 相关的导入 ---
from torch_geometric.nn import GATv2Conv, global_mean_pool
# --- 结束添加 ---

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers import (
    AutoConfig,
    AutoModel,
)
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPooling
)
from transformers.models.roberta.modeling_roberta import (
    RobertaOutput
)
from . import LongelmConfig, LongelmModel
from .configuration_dualencoder import DualEncoderConfig


logger = logging.get_logger(__name__)

# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class CASPOutput(ModelOutput):  # TODO: modify this part later
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_source:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_assembly:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        source_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The source code embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        assembly_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The assembly embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        source_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        assembly_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_source: torch.FloatTensor = None
    logits_per_assembly: torch.FloatTensor = None
    source_embeds: torch.FloatTensor = None
    assembly_embeds: torch.FloatTensor = None
    source_model_output: BaseModelOutputWithPooling = None
    assembly_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["source_model_output", "assembly_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )

# --- 2. 添加 GNN 模块定义 ---
class CG_GNN_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(input_dim, hidden_dim, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads))
        
        # GNN 的输出维度将是 hidden_dim * heads
        self.output_dim = hidden_dim * heads

    def forward(self, x, edge_index):
        # x: [N_total_nodes, input_dim] (来自 Longelm 的初始特征)
        # edge_index: [2, Num_Edges]
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # 最后一层之前应用激活和 dropout
            if i < len(self.convs) - 1: 
                x = nn.functional.elu(x)
                x = nn.functional.dropout(x, p=0.1, training=self.training)
            
        # 返回所有节点强化后的特征
        # x shape: [N_total_nodes, hidden_dim * heads]
        return x
# --- 结束添加 ---

class DualEncoderModel(PreTrainedModel):
    config_class = DualEncoderConfig
    base_model_prefix = 'dual_encoder'

    def __init__(
        self,
        config: Optional[DualEncoderConfig] = None,
        assembly_model: Optional[PreTrainedModel] = None,
        source_model: Optional[PreTrainedModel] = None,
    ):
        if config is None and (assembly_model is None or source_model is None):
            raise ValueError("Either a configuration or an assembly model and a text model has to be provided")

        if config is None:
            config = DualEncoderConfig.from_assembly_source_configs(assembly_model.config, source_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise  ValueError(f"config: {config} has to be of type {self.config_class}")
            
        # initialize with config
        super().__init__(config)

        if assembly_model is None:
            config.assembly_config = LongelmConfig.from_dict(config.assembly_config)
            assembly_model = LongelmModel(config.assembly_config)
        
        if source_model is None:
            source_model = AutoModel.from_config(config.source_config)

        self.assembly_model = assembly_model
        # NOTE: remove projection head of codet5p-110m-embedding model (temp)
        if hasattr(source_model, 'encoder') and hasattr(source_model, 'proj'):
            self.source_model = source_model.encoder
        else:
            self.source_model = source_model

        # make sure that the individual model's config refers to the shared config
        # so that the udpates to the config will be synced
        self.assembly_model.config = self.config.assembly_config
        self.source_model.config = self.config.source_config

        self.assembly_embed_dim = config.assembly_config.hidden_size
        self.source_embed_dim = config.source_config.hidden_size
        self.projection_dim = config.projection_dim

        # --- 3. 修改 __init__ 以添加 GNN ---
        
        # 3.1 实例化 GNN 模块
        # (您可以将 gnn_hidden_dim 和 heads 添加到 config 中，这里使用硬编码示例)
        gnn_hidden_dim = self.assembly_embed_dim // 4 
        self.gnn_encoder = CG_GNN_Encoder(
            input_dim=self.assembly_embed_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=2,
            heads=4
        )
        
        # 3.2 修改投影层以匹配 GNN 的输出
        # self.assembly_projection = nn.Linear(self.assembly_embed_dim, self.projection_dim, bias=False)
        self.assembly_projection = nn.Linear(
            self.gnn_encoder.output_dim, 
            self.projection_dim, 
            bias=False
        )
        # --- 结束修改 ---

        self.source_projection = nn.Linear(self.source_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

    def get_source_features(
        self,
        input_ids=None,
        attention_mask=None,
        # position_ids=None,
        # token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            source_features (`torch.FloatTensor` of shape `(batch_size, output_dim)`)
        """
        source_outputs = self.source_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            # token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = source_outputs[0][:, 0, :]
        source_features = self.source_projection(pooled_output)

        return source_features
    
    def get_assembly_features(
        self,
        input_ids=None,
        attention_mask=None,
        graph_attention_mask=None,
        relative_node_positions=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            assembly_features (`torch.FloatTensor` of shape `(batch_size, output_dim)`)
        """
        assembly_output = self.assembly_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_attention_mask=graph_attention_mask,
            relative_node_positions=relative_node_positions,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_outputs = assembly_output[1]
        assembly_features = self.assembly_projection(pooled_outputs)

        return assembly_features
    
    def forward(
        self,
        source_input_ids: Optional[torch.LongTensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        # source_position_ids: Optional[torch.LongTensor] = None,
        # source_token_type_ids: Optional[torch.LongTensor] = None,

        # --- 4. 修改 forward 签名 ---
        # (删除旧的 assembly_... 参数)
        # assembly_input_ids: Optional[torch.LongTensor] = None,
        # assembly_attention_mask: Optional[torch.Tensor] = None,
        # assembly_graph_attention_mask: Optional[torch.Tensor] = None,
        # assembly_relative_node_positions: Optional[torch.LongTensor] = None,

        # (添加新的 longelm_... 和 gnn_... 参数)
        longelm_input_ids: Optional[torch.LongTensor] = None,
        longelm_attention_mask: Optional[torch.Tensor] = None,
        longelm_graph_attention_mask: Optional[torch.Tensor] = None,
        longelm_relative_node_positions: Optional[torch.LongTensor] = None,
        
        gnn_edge_index: Optional[torch.LongTensor] = None,
        gnn_batch_index: Optional[torch.LongTensor] = None,
        gnn_target_node_indices: Optional[torch.LongTensor] = None,
        # --- 结束修改签名 ---

        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        return_loss: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CASPOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        source_outputs = self.source_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            # token_type_ids=source_token_type_ids,
            # position_ids=source_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # --- 5. 修改汇编侧逻辑 ---
        # (删除旧的 assembly_outputs 计算)
        # assembly_outputs = self.assembly_model( ... )

        # 步骤 1: Longelm (CodeArt) 对所有节点进行函数内编码
        longelm_outputs = self.assembly_model(
            input_ids=longelm_input_ids,
            attention_mask=longelm_attention_mask,
            graph_attention_mask=longelm_graph_attention_mask,
            relative_node_positions=longelm_relative_node_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 提取所有 N_total_nodes 个节点的初始特征
        # 假设 LongelmModel 的 pooler_output 是元组的第二个元素
        initial_node_features = longelm_outputs[1] # [N_total_nodes, longelm_hidden_dim]
        
        # 步骤 2: GNN 模块进行上下文强化
        reinforced_node_features = self.gnn_encoder(
            x=initial_node_features,
            edge_index=gnn_edge_index
        ) # [N_total_nodes, gnn_output_dim]
        
        # 步骤 3: 提取 Target 节点的特征
        target_features = reinforced_node_features.index_select(0, gnn_target_node_indices)
        # [Batch_size, gnn_output_dim]
        
        # 步骤 4: 投影
        assembly_embeds = self.assembly_projection(target_features)
        # --- 结束修改汇编侧 ---

        # assembly_outputs = self.assembly_model(
        #     input_ids=assembly_input_ids,
        #     attention_mask=assembly_attention_mask,
        #     graph_attention_mask=assembly_graph_attention_mask,
        #     relative_node_positions=assembly_relative_node_positions,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        # NOTE: brute-force pooling for T5Encoder
        # source_embeds = source_outputs[1]
        source_embeds = source_outputs[0][:, 0, :]
        source_embeds = self.source_projection(source_embeds)

        # assembly_embeds = assembly_outputs[1]
        # assembly_embeds = self.assembly_projection(assembly_embeds)

        # normalize features
        z1 = source_embeds / source_embeds.norm(dim=-1, keepdim=True)
        z2 = assembly_embeds / assembly_embeds.norm(dim=-1, keepdim=True)

        if dist.is_initialized() and self.training:
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # logits_per_source = torch.matmul(source_embeds, assembly_embeds.t()) * logit_scale
        logits_per_source = torch.matmul(z1, z2.t()) * logit_scale
        logits_per_assembly = logits_per_source.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_source)

        if not return_dict:
            output = (logits_per_source, logits_per_assembly, source_embeds, assembly_embeds, source_outputs, longelm_outputs)
            return ((loss,) + output) if loss is not None else output
        
        return CASPOutput(
            loss=loss,
            logits_per_source=logits_per_source,
            logits_per_assembly=logits_per_assembly,
            source_embeds=source_embeds,
            assembly_embeds=assembly_embeds,
            source_model_output=source_outputs,
            assembly_model_output=longelm_outputs
        )
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)


class MomentumDualEncoderModel(DualEncoderModel):

    def __init__(
        self,
        config: Optional[DualEncoderConfig] = None,
        assembly_model: Optional[PreTrainedModel] = None,
        source_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(
            config=config,
            assembly_model=assembly_model,
            source_model=source_model
        )
        self.K = config.K
        self.m = config.m  # TODO: different m for each encoder

        self.assembly_model_k = deepcopy(self.assembly_model)
        for param_k in self.assembly_model_k.parameters():
            param_k.requires_grad = False
        self.source_model_k = deepcopy(self.source_model)
        for param_k in self.source_model_k.parameters():
            param_k.requires_grad = False
        
        # --- 6. 为 GNN 添加 Key 副本 (投影层保持共享) ---
        self.gnn_encoder_k = deepcopy(self.gnn_encoder)
        for param_k in self.gnn_encoder_k.parameters():
            param_k.requires_grad = False
        # --- 结束添加 ---

        # create the queues
        self.register_buffer("source_queue", torch.randn(config.projection_dim, self.K))
        self.source_queue = nn.functional.normalize(self.source_queue, dim=0)
        self.register_buffer("assembly_queue", torch.randn(config.projection_dim, self.K))
        self.assembly_queue = nn.functional.normalize(self.assembly_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_unpdate_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # update source model
        for param_q, param_k in zip(
            self.source_model.parameters(), self.source_model_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        
        # update assembly model
        for param_q, param_k in zip(
            self.assembly_model.parameters(), self.assembly_model_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
            
        # --- 7. 添加 GNN 的动量更新 ---
        for param_q, param_k in zip(
            self.gnn_encoder.parameters(), self.gnn_encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        # --- 结束添加 ---

    @torch.no_grad()
    def _dequeue_and_enqueue(self, source_keys, assembly_keys):
        # gather keys before updating queue
        source_keys = concat_all_gather(source_keys)
        assembly_keys = concat_all_gather(assembly_keys)
    
        batch_size = source_keys.shape[0]
        K = self.K
        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # --- 关键改动，从这里只写得下的部分 ---
        if ptr + batch_size <= K:
            # 一次写完
            self.source_queue[:, ptr:ptr + batch_size] = source_keys.T
            self.assembly_queue[:, ptr:ptr + batch_size] = assembly_keys.T
            ptr = (ptr + batch_size) % K
        else:
            # 队尾写一部分，队首写一部分
            end_len = K - ptr
            rest_len = batch_size - end_len
            # 队尾
            self.source_queue[:, ptr:] = source_keys[:end_len].T
            self.assembly_queue[:, ptr:] = assembly_keys[:end_len].T
            # 队首
            self.source_queue[:, :rest_len] = source_keys[end_len:].T
            self.assembly_queue[:, :rest_len] = assembly_keys[end_len:].T
            ptr = rest_len  # 已经循环

        self.queue_ptr[0] = ptr

    def forward(
        self,
        source_input_ids: Optional[torch.LongTensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        # source_position_ids: Optional[torch.LongTensor] = None,
        # source_token_type_ids: Optional[torch.LongTensor] = None,

        # --- 8. 修改 forward 签名 (同上) ---
        longelm_input_ids: Optional[torch.LongTensor] = None,
        longelm_attention_mask: Optional[torch.Tensor] = None,
        longelm_graph_attention_mask: Optional[torch.Tensor] = None,
        longelm_relative_node_positions: Optional[torch.LongTensor] = None,
        
        gnn_edge_index: Optional[torch.LongTensor] = None,
        gnn_batch_index: Optional[torch.LongTensor] = None,
        gnn_target_node_indices: Optional[torch.LongTensor] = None,
        
        # (删除旧的 assembly_... 参数)
        # assembly_input_ids: Optional[torch.LongTensor] = None,
        # ...
        # --- 结束修改签名 ---

        # assembly_input_ids: Optional[torch.LongTensor] = None,
        # assembly_attention_mask: Optional[torch.Tensor] = None,
        # assembly_graph_attention_mask: Optional[torch.Tensor] = None,
        # assembly_relative_node_positions: Optional[torch.LongTensor] = None,

        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        return_loss: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CASPOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        source_outputs = self.source_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            # token_type_ids=source_token_type_ids,
            # position_ids=source_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # --- 9. 修改汇编侧 Query (q2) ---
        # (复制 DualEncoderModel.forward 中的 步骤 1-4)
        longelm_outputs_q = self.assembly_model(
            input_ids=longelm_input_ids,
            attention_mask=longelm_attention_mask,
            graph_attention_mask=longelm_graph_attention_mask,
            relative_node_positions=longelm_relative_node_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        initial_node_features_q = longelm_outputs_q[1]
        reinforced_node_features_q = self.gnn_encoder(
            x=initial_node_features_q,
            edge_index=gnn_edge_index
        )
        target_features_q = reinforced_node_features_q.index_select(0, gnn_target_node_indices)
        assembly_embeds = self.assembly_projection(target_features_q)
        # --- 结束修改 q2 ---

        # assembly_outputs = self.assembly_model(
        #     input_ids=assembly_input_ids,
        #     attention_mask=assembly_attention_mask,
        #     graph_attention_mask=assembly_graph_attention_mask,
        #     relative_node_positions=assembly_relative_node_positions,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        # NOTE: brute-force pooling for T5Encoder
        # source_embeds = source_outputs[1]
        source_embeds = source_outputs[0][:, 0, :]
        source_embeds = self.source_projection(source_embeds)

        # assembly_embeds = assembly_outputs[1]
        # assembly_embeds = self.assembly_projection(assembly_embeds)

        # normalize features
        q1 = source_embeds / source_embeds.norm(dim=-1, keepdim=True)
        q2 = assembly_embeds / assembly_embeds.norm(dim=-1, keepdim=True)

        # --- 10. 修改汇编侧 Key (k2) ---
        with torch.no_grad():
            if self.training:
                self._momentum_unpdate_key_encoder() # (这个函数已被我们更新)

            # 源码侧 Key (k1) (保持不变)
            k1 = self.source_model_k(
                input_ids=source_input_ids,
                # ...
                return_dict=return_dict,
            )[0][:, 0, :]
            k1 = self.source_projection(k1) # 遵循原始代码，使用共享投影层
            k1 = k1 / k1.norm(dim=-1, keepdim=True)

            # 汇编侧 Key (k2)
            # (删除旧的 k2 计算)
            # k2 = self.assembly_model_k(...)
            # k2 = self.assembly_projection(k2)

            # (使用 CodeArt -> GNN -> 投影 流程，但使用 _k 模型)
            longelm_outputs_k = self.assembly_model_k(
                input_ids=longelm_input_ids,
                attention_mask=longelm_attention_mask,
                graph_attention_mask=longelm_graph_attention_mask,
                relative_node_positions=longelm_relative_node_positions,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            initial_node_features_k = longelm_outputs_k[1]
            reinforced_node_features_k = self.gnn_encoder_k(
                x=initial_node_features_k,
                edge_index=gnn_edge_index
            )
            target_features_k = reinforced_node_features_k.index_select(0, gnn_target_node_indices)
            k2 = self.assembly_projection(target_features_k) # 遵循原始代码，使用共享投影层
            k2 = k2 / k2.norm(dim=-1, keepdim=True)
            # --- 结束修改 k2 ---

        # positive logits
        src_l_pos = torch.einsum("nc,nc->n", [q1, k2]).unsqueeze(-1)
        asm_l_pos = torch.einsum("nc,nc->n", [q2, k1]).unsqueeze(-1)
        # negative logits
        src_l_neg = torch.einsum("nc,ck->nk", [q1, self.assembly_queue.clone().detach()])
        asm_l_neg = torch.einsum("nc,ck->nk", [q2, self.source_queue.clone().detach()])

        # logits: Nx(1+K)
        src_logits = torch.cat([src_l_pos, src_l_neg], dim=1)
        asm_logits = torch.cat([asm_l_pos, asm_l_neg], dim=1)

        # apply scale
        logit_scale = self.logit_scale.exp()
        src_logits = logit_scale * src_logits
        asm_logits = logit_scale * asm_logits

        # labels: positive key indicators
        src_labels = torch.zeros(src_logits.shape[0], dtype=torch.long).cuda()
        asm_labels = torch.zeros(asm_logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if self.training:
            self._dequeue_and_enqueue(
                source_keys=k1, 
                assembly_keys=k2
            )

        # compute loss
        src_loss = nn.functional.cross_entropy(src_logits, src_labels)
        asm_loss = nn.functional.cross_entropy(asm_logits, asm_labels)
        loss = (src_loss + asm_loss) / 2.0

        return CASPOutput(
            loss=loss,
            logits_per_source=src_logits,
            logits_per_assembly=asm_logits,
            source_embeds=k1,
            assembly_embeds=k2,
        )


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class DualEncoderRanker(MomentumDualEncoderModel):

    def forward(
        self,
        source_input_ids: Optional[torch.LongTensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,

        # assembly_input_ids: Optional[torch.LongTensor] = None,
        # assembly_attention_mask: Optional[torch.Tensor] = None,
        # assembly_graph_attention_mask: Optional[torch.Tensor] = None,
        # assembly_relative_node_positions: Optional[torch.LongTensor] = None,

        # --- 修改输入签名 ---
        longelm_input_ids: Optional[torch.LongTensor] = None,
        longelm_attention_mask: Optional[torch.Tensor] = None,
        longelm_graph_attention_mask: Optional[torch.Tensor] = None,
        longelm_relative_node_positions: Optional[torch.LongTensor] = None,
        
        gnn_edge_index: Optional[torch.LongTensor] = None,
        gnn_batch_index: Optional[torch.LongTensor] = None,
        gnn_target_node_indices: Optional[torch.LongTensor] = None,
        # ------------------

        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        target_scores: Optional[torch.FloatTensor] = None,
        log_target: Optional[bool] = False,

        return_loss: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CASPOutput]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        source_outputs = self.source_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            # token_type_ids=source_token_type_ids,
            # position_ids=source_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # assembly_outputs = self.assembly_model(
        #     input_ids=assembly_input_ids,
        #     attention_mask=assembly_attention_mask,
        #     graph_attention_mask=assembly_graph_attention_mask,
        #     relative_node_positions=assembly_relative_node_positions,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        longelm_outputs = self.assembly_model(
            input_ids=longelm_input_ids,
            attention_mask=longelm_attention_mask,
            graph_attention_mask=longelm_graph_attention_mask,
            relative_node_positions=longelm_relative_node_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # NOTE: brute-force pooling for T5Encoder
        source_embeds = source_outputs[0][:, 0, :]
        source_embeds = self.source_projection(source_embeds)

        initial_node_features = longelm_outputs[1]
        
        reinforced_node_features = self.gnn_encoder(
            x=initial_node_features,
            edge_index=gnn_edge_index
        )
        
        target_features = reinforced_node_features.index_select(0, gnn_target_node_indices)
        assembly_embeds = self.assembly_projection(target_features)

        # normalize features
        e_src = source_embeds / source_embeds.norm(dim=-1, keepdim=True)
        e_asm = assembly_embeds / assembly_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        k = target_scores.shape[1]  # e_asm: (batch_size, embed_dim), e_src: (batch_size * k, embed_dim)
        e_src = e_src.view(e_asm.shape[0], k, e_asm.shape[1])   # (batch_size, k, embed_dim)
        logits_per_asm = torch.matmul(e_asm[:, None, :], e_src.transpose(1, 2)).\
                                                        squeeze() * logit_scale
        log_input = nn.functional.log_softmax(logits_per_asm, dim=1)

        # shape correctness
        assert(log_input.shape == target_scores.shape)

        loss = None
        if return_loss:
            loss = nn.functional.kl_div(log_input, target_scores, log_target=log_target)   # NOTE: log_target is False by default

        if not return_dict:
            output = (None, logits_per_asm, source_embeds, assembly_embeds, source_outputs, longelm_outputs)
            return ((loss,) + output) if loss is not None else output
        
        return CASPOutput(
            loss=loss,
            logits_per_source=None,
            logits_per_assembly=logits_per_asm,
            source_embeds=source_embeds,
            assembly_embeds=assembly_embeds,
            source_model_output=source_outputs,
            assembly_model_output=longelm_outputs
        )