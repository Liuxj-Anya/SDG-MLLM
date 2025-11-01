import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedModel, PretrainedConfig
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import HeteroConv, GATConv
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

from torch_geometric.data import HeteroData

def build_hetero_graph(node_feats, edge_index, edge_type, rel_id2name, device):
    data = HeteroData()
    data['token'].x = node_feats  # [N, D]

    for rel_id, rel_name in rel_id2name.items():
        mask = (edge_type == rel_id)
        ei = edge_index[:, mask]
        if ei.size(1) > 0:  # 只添加存在的边
            data['token', rel_name, 'token'].edge_index = ei

    return data.to(device)

class HGTEncoder(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, metadata, num_heads=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads
            )
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return x_dict


class TextImageConfig(PretrainedConfig):
    model_type = "text-image-causal-lm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_model_path = kwargs.get("text_model_path", "")
        self.image_feature_dim = kwargs.get("image_feature_dim", 1536)
        self.audio_feature_dim = kwargs.get("audio_feature_dim", 768)

class Qwen3_HGTEncoder(PreTrainedModel):
    config_class = TextImageConfig

    def __init__(self, config, lora_config=None, tokenizer=None, gcn_layers=2, rel_id2name=None):
        super().__init__(config)

        self.text_model = AutoModelForCausalLM.from_pretrained(
            config.text_model_path,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = tokenizer
        self.text_model.resize_token_embeddings(len(self.tokenizer))
        self.embedding_dim = self.text_model.config.hidden_size
        self.pad_token_id = self.tokenizer.pad_token_id

        # for p in self.text_model.get_input_embeddings().parameters():
        #     p.requires_grad = False

        if lora_config is not None:
            self.text_model = get_peft_model(self.text_model, lora_config)

        image_feature_dim = config.image_feature_dim
        audio_feature_dim = config.audio_feature_dim
        text_embedding_dim = self.text_model.config.hidden_size
        # 图像投影层，并放到与主模型同一设备
        # self.image_projector = nn.Linear(image_feature_dim, text_embedding_dim)
        # self.audio_projector = nn.Linear(audio_feature_dim, text_embedding_dim)
        # self.vid_projector = nn.Linear(image_feature_dim, text_embedding_dim)
        self.image_projector = nn.Sequential(
            nn.Linear(image_feature_dim, text_embedding_dim),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(text_embedding_dim, text_embedding_dim),
            nn.Dropout(p=0.1)
        )
        self.audio_projector = nn.Sequential(
            nn.Linear(audio_feature_dim, text_embedding_dim),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(text_embedding_dim, text_embedding_dim),
            nn.Dropout(p=0.1)
        )
        self.vid_projector = nn.Sequential(
            nn.Linear(image_feature_dim, text_embedding_dim),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(text_embedding_dim, text_embedding_dim),
            nn.Dropout(p=0.1)
        )
        # 冻结除特殊 token 外的 embedding
        # 独立 embedding，只处理 <img> 和 </img>
        self.special_token_embed = nn.Embedding(11, text_embedding_dim)  # index 0: <img>, 1: </img>
        # 保存 token id 映射
        self.img_token_id = self.tokenizer.convert_tokens_to_ids('<img>')
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids('</img>')
        with torch.no_grad():
            self.special_token_embed.weight[0] = self.text_model.get_input_embeddings().weight[self.img_token_id]
            self.special_token_embed.weight[1] = self.text_model.get_input_embeddings().weight[self.img_end_token_id]

        # 3. 输出融合回原维度
        self.rel_id2name = rel_id2name  # dict: {0: 'dep', 1: 'coref', ...}

        self.gcn_layers = gcn_layers

        node_types = ['token']
        edge_types = [('token', rel_name, 'token') for rel_name in self.rel_id2name.values()]
        metadata = (node_types, edge_types)

        self.hgt_encoder = HGTEncoder(
            hidden_dim=self.embedding_dim,  # same as input embedding dim
            num_layers=self.gcn_layers,  # reuse GCN layer count
            metadata=(['token'], [
                ('token', rel_name, 'token')
                for rel_name in self.rel_id2name.values()
            ]),
            num_heads=2  # you can tune this
        )

        self.gate_layer = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()  # 输出 α ∈ (0,1)，shape: [L, 1]
        )

    def _move_to_first_device(self, module):
        if hasattr(self.text_model, 'hf_device_map'):
            first_module_name = list(self.text_model.hf_device_map.keys())[0]
            target_device = self.text_model.hf_device_map[first_module_name]
            return module.to(target_device)
        else:
            return module.to('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_edge_index_dict(self, edge_index, edge_type, device):
        edge_dict = defaultdict(list)
        for i in range(edge_type.size(0)):
            rel_id = edge_type[i].item()
            rel_name = self.rel_id2name[rel_id]
            src = edge_index[0, i].item()
            tgt = edge_index[1, i].item()
            edge_dict[rel_name].append((src, tgt))

        edge_index_dict = {}
        for rel in self.rel_id2name.values():
            edges = edge_dict.get(rel, [])
            edge_index_tensor = (
                torch.tensor(edges, dtype=torch.long, device=device).T
                if edges else torch.empty((2, 0), dtype=torch.long, device=device)
            )
            # 修改点：使用三元组作为 key
            edge_index_dict[('node', rel, 'node')] = edge_index_tensor

        return edge_index_dict

    def generate_with_image(self, input_ids, attention_mask=None, edge_index=None, edge_type=None, graph_spans=None,
                            utterance_ranges=None, image_feature=None, audio_feature=None, vid_feature=None,
                            image_tag=None, audio_tag=None, vid_tag=None, task_id=None, generation_kwargs=None):
        self.eval()

        generation_kwargs = generation_kwargs or {}
        device = self.image_projector[0].weight.device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        graph_spans = graph_spans.to(device)
        utterance_ranges = utterance_ranges.to(device)
        task_id.to(device)
        B, L = input_ids.size()
        self.special_token_embed.to(device)
        self.hgt_encoder.to(device)
        self.text_model.to(device)
        self.gate_layer.to(device)
        # 用独立的 embedding 生成特殊 token embedding
        img_token_embed = self.special_token_embed(torch.tensor([0], device=device)).unsqueeze(0).expand(B, 1, -1)
        img_end_token_embed = self.special_token_embed(torch.tensor([1], device=device)).unsqueeze(0).expand(B, 1, -1)

        audio_token_embed = self.special_token_embed(torch.tensor([2], device=device)).unsqueeze(0).expand(B, 1, -1)
        audio_end_token_embed = self.special_token_embed(torch.tensor([3], device=device)).unsqueeze(0).expand(B, 1, -1)

        vid_token_embed = self.special_token_embed(torch.tensor([4], device=device)).unsqueeze(0).expand(B, 1, -1)
        vid_end_token_embed = self.special_token_embed(torch.tensor([5], device=device)).unsqueeze(0).expand(B, 1, -1)

        no_img_token_embed = self.special_token_embed(torch.tensor([6], device=device)).unsqueeze(0).expand(B, 1, -1)
        no_aud_token_embed = self.special_token_embed(torch.tensor([7], device=device)).unsqueeze(0).expand(B, 1, -1)
        no_vid_token_embed = self.special_token_embed(torch.tensor([8], device=device)).unsqueeze(0).expand(B, 1, -1)

        task_id_token_embed = self.special_token_embed((task_id + 9).to(device)).unsqueeze(0).expand(B, 1, -1)

        # 1. 获取原始 token embedding
        input_embeds = self.text_model.get_input_embeddings()(input_ids)  # [B, L, D]
        B, L, D = input_embeds.shape

        # 2. 收集 graph token embedding 用作 GCN 输入
        graph_node_feats = []
        graph_token_counts = []

        for i in range(B):
            start, end = graph_spans[i]
            node_feats = input_embeds[i, start:end]  # [num_node_i, D]
            graph_node_feats.append(node_feats)
            graph_token_counts.append(end - start)

        graph_node_feats = torch.cat(graph_node_feats, dim=0).to(torch.float32)  # [total_node_num, D]

        # 3. GCN 编码
        #gcn_out = self.gcn(graph_node_feats, edge_index, edge_type)  # [total_node_num, D]
        hetero_data = build_hetero_graph(graph_node_feats, edge_index, edge_type, self.rel_id2name,
                                         device=input_ids.device)
        hgt_out_dict = self.hgt_encoder(hetero_data.x_dict, hetero_data.edge_index_dict)
        gcn_out = hgt_out_dict['token']  # [total_node_num, D]

        # 4. 将 GCN 输出按样本还原，融合回 input_embeds
        new_input_embeds = input_embeds.clone()

        cur = 0
        for i in range(B):
            start, end = graph_spans[i]
            length = end - start

            gcn_feat = gcn_out[cur:cur + length]  # [length, D]
            orig_feat = input_embeds[i, start:end]  # [length, D]
            # [L, D] -> 拼接后 [L, 3D]
            gate_input = torch.cat([orig_feat, gcn_feat], dim=-1)
            alpha = self.gate_layer(gate_input)  # [L, D]
            gate = alpha.mean(dim=-2, keepdim=True)
            fused_feat = (1 - gate) * orig_feat + gate * gcn_feat  # [L, D]
            #fused_feat = (1 - self.fusion_alpha) * orig_feat + self.fusion_alpha * gcn_feat
            #input_embeds[i, start:end] = fused_feat  # inplace 替换
            new_input_embeds[i, start:end] = fused_feat

            cur += length

        gcn_embeds = new_input_embeds.clone()

        new_embeddings = []
        new_attn_masks = []
        new_labels = []

        utterance_ranges[0][0] = utterance_ranges[0][0] - 1

        for i in range(B):
            emb = gcn_embeds[i]  # (L, H)
            att_mask = attention_mask[i] if attention_mask is not None else None
            u_ranges = utterance_ranges

            cur_embeds = [emb[0:u_ranges[0][0]]]
            cur_mask = [torch.ones(emb[0:u_ranges[0][0]].size(0), dtype=emb.dtype).to(emb.device)]

            cur_embeds.append(task_id_token_embed[0].to(emb.device))
            cur_mask.append(torch.ones(task_id_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

            for ui, (start, end) in enumerate(u_ranges):
                # 插入原始 embedding 区段
                cur_embeds.append(emb[start:end])
                cur_mask.append(torch.ones(emb[start:end].size(0), dtype=emb.dtype).to(emb.device))

                # 判断是否全为0
                image_all_zero = image_tag[i][ui] == 0
                if image_all_zero == False:
                    # 插入图像摘要 embedding（每个 utterance 后面插入
                    img_np = torch.from_numpy(
                        np.load('/data2/liuxj/1-mcabsa/test/mod_feature/img/' + image_feature[i][ui] + '.npy')).to(
                        device)
                    img_feat = self.image_projector(img_np)  # [1, hidden]
                    cur_embeds.append(img_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(img_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(img_feat[0].to(emb.device))
                    cur_mask.append(torch.ones(img_feat[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(img_end_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(img_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                else:
                    cur_embeds.append(img_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(img_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(no_img_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(no_img_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(img_end_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(img_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                # 判断是否全为0
                audio_all_zero = audio_tag[i][ui] == 0
                if audio_all_zero == False:
                    aud_np = torch.from_numpy(
                        np.load('/data2/liuxj/1-mcabsa/test/mod_feature/aud/' + audio_feature[i][ui] + '.npy')).to(
                        device)
                    aud_feat = self.audio_projector(aud_np)

                    cur_embeds.append(audio_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(audio_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(aud_feat[0].to(emb.device))
                    cur_mask.append(torch.ones(aud_feat[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(audio_end_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(audio_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                else:
                    cur_embeds.append(audio_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(audio_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(no_aud_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(no_aud_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(audio_end_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(audio_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                    # 判断是否全为0
                vid_all_zero = vid_tag[i][ui] == 0
                if vid_all_zero == False:
                    vid_np = torch.from_numpy(
                        np.load('/data2/liuxj/1-mcabsa/test/mod_feature/vid/' + vid_feature[i][ui] + '.npy')).unsqueeze(
                        0).to(device)
                    vid_feat = self.vid_projector(vid_np)

                    cur_embeds.append(vid_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(vid_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(vid_feat[0].to(emb.device))
                    cur_mask.append(torch.ones(vid_feat[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(vid_end_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(vid_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                else:
                    cur_embeds.append(vid_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(vid_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(no_vid_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(no_vid_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

                    cur_embeds.append(vid_end_token_embed[0].to(emb.device))
                    cur_mask.append(torch.ones(vid_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))

            # 拼接 utterance+image
            new_embeddings.append(torch.cat(cur_embeds, dim=0))  # (L + num_utterance, H)
            new_attn_masks.append(torch.cat(cur_mask, dim=0))

        # 对所有 batch 内 pad 到相同长度
        gcn_embeds = pad_sequence(new_embeddings, batch_first=True, padding_value=self.pad_token_id).to(
            torch.bfloat16)  # (B, L', H)
        if attention_mask is not None:
            attention_mask = pad_sequence(new_attn_masks, batch_first=True, padding_value=0).to(torch.bfloat16)

        # 使用 generate（注意我们传入的是 inputs_embeds）
        outputs = self.text_model.generate(
            inputs_embeds=gcn_embeds,
            attention_mask=attention_mask,
            **generation_kwargs
        )

        return outputs

    def forward(self, input_ids, attention_mask=None, labels=None, edge_index=None, edge_type=None, graph_spans=None,
                utterance_ranges=None, image_feature=None, audio_feature=None, vid_feature=None, image_tag=None,
                audio_tag=None, vid_tag=None, task_id=None):
        """
        input_ids: (B, L) LongTensor
        edge_index: (2, E) LongTensor
        edge_type: (E,) LongTensor, 每条边的类型 ID（如 dep=0, coref=1,...）
        """

        device = input_ids.device
        B, L = input_ids.size()
        # 用独立的 embedding 生成特殊 token embedding
        img_token_embed = self.special_token_embed(torch.tensor([0], device=device)).unsqueeze(0).expand(B, 1, -1)
        img_end_token_embed = self.special_token_embed(torch.tensor([1], device=device)).unsqueeze(0).expand(B, 1, -1)

        audio_token_embed = self.special_token_embed(torch.tensor([2], device=device)).unsqueeze(0).expand(B, 1, -1)
        audio_end_token_embed = self.special_token_embed(torch.tensor([3], device=device)).unsqueeze(0).expand(B, 1, -1)

        vid_token_embed = self.special_token_embed(torch.tensor([4], device=device)).unsqueeze(0).expand(B, 1, -1)
        vid_end_token_embed = self.special_token_embed(torch.tensor([5], device=device)).unsqueeze(0).expand(B, 1, -1)

        no_img_token_embed = self.special_token_embed(torch.tensor([6], device=device)).unsqueeze(0).expand(B, 1, -1)
        no_aud_token_embed = self.special_token_embed(torch.tensor([7], device=device)).unsqueeze(0).expand(B, 1, -1)
        no_vid_token_embed = self.special_token_embed(torch.tensor([8], device=device)).unsqueeze(0).expand(B, 1, -1)

        task_id_token_embed = self.special_token_embed(task_id + 9).unsqueeze(0).expand(B, 1, -1)

        # 1. 获取原始 token embedding
        input_embeds = self.text_model.get_input_embeddings()(input_ids)  # [B, L, D]
        B, L, D = input_embeds.shape

        # 2. 收集 graph token embedding 用作 GCN 输入
        graph_node_feats = []
        graph_token_counts = []

        for i in range(B):
            start, end = graph_spans[i]
            node_feats = input_embeds[i, start:end]  # [num_node_i, D]
            graph_node_feats.append(node_feats)
            graph_token_counts.append(end - start)

        graph_node_feats = torch.cat(graph_node_feats, dim=0)  # [total_node_num, D]

        # 3. GCN 编码
        #gcn_out = self.gcn(graph_node_feats, edge_index, edge_type)  # [total_node_num, D]
        hetero_data = build_hetero_graph(graph_node_feats, edge_index, edge_type, self.rel_id2name,
                                         device=input_ids.device)
        hgt_out_dict = self.hgt_encoder(hetero_data.x_dict, hetero_data.edge_index_dict)
        gcn_out = hgt_out_dict['token']  # [total_node_num, D]

        # 4. 将 GCN 输出按样本还原，融合回 input_embeds
        new_input_embeds = input_embeds.clone()
        cur = 0
        for i in range(B):
            start, end = graph_spans[i]
            length = end - start

            gcn_feat = gcn_out[cur:cur + length]  # [length, D]
            orig_feat = input_embeds[i, start:end]  # [length, D]
            # [L, D] -> 拼接后 [L, 3D]
            gate_input = torch.cat([orig_feat, gcn_feat], dim=-1)
            alpha = self.gate_layer(gate_input)  # [L, D]
            gate = alpha.mean(dim=-2, keepdim=True)
            fused_feat = (1 - gate) * orig_feat + gate * gcn_feat  # [L, D]
            #fused_feat = (1 - self.fusion_alpha) * orig_feat + self.fusion_alpha * gcn_feat
            #input_embeds[i, start:end] = fused_feat  # inplace 替换
            new_input_embeds[i, start:end] = fused_feat

            cur += length

        gcn_embeds = new_input_embeds.clone()

        # new_embeddings = []
        # new_attn_masks = []
        # new_labels = []
        #
        # utterance_ranges[0][0] = utterance_ranges[0][0] - 1
        #
        # for i in range(B):
        #     emb = gcn_embeds[i]  # (L, H)
        #     att_mask = attention_mask[i] if attention_mask is not None else None
        #     label = labels[i] if labels is not None else None
        #
        #     # 方法：使用布尔掩码
        #     lmask = label != -100  # 获取一个布尔掩码
        #     label = label[lmask]  # 只保留不为 -100 的元素
        #
        #     u_ranges = utterance_ranges
        #
        #     cur_embeds = [emb[0:u_ranges[0][0]]]
        #     cur_mask = [torch.ones(emb[0:u_ranges[0][0]].size(0), dtype=emb.dtype).to(emb.device)]
        #     cur_label = [torch.full((emb[0:u_ranges[0][0]].size(0),), -100, dtype=labels.dtype, device=labels.device)]
        #
        #     cur_embeds.append(task_id_token_embed[0].to(emb.device))
        #     cur_mask.append(torch.ones(task_id_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #     cur_label.append(
        #         torch.full((task_id_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #     for ui, (start, end) in enumerate(u_ranges):
        #         # 插入原始 embedding 区段
        #         cur_embeds.append(emb[start:end])
        #         cur_mask.append(torch.ones(emb[start:end].size(0), dtype=emb.dtype).to(emb.device))
        #         cur_label.append(torch.full((emb[start:end].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #         # 判断是否全为0
        #         image_all_zero = image_tag[i][ui] == 0
        #         if image_all_zero == False:
        #             # 插入图像摘要 embedding（每个 utterance 后面插入
        #             img_np = torch.from_numpy(np.load('data/mod_feature/img/' + image_feature[i][ui] + '.npy')).to(
        #                 device)
        #             img_feat = self.image_projector(img_np.to(torch.bfloat16))  # [1, hidden]
        #             cur_embeds.append(img_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(img_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((img_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             cur_embeds.append(img_feat[0].to(emb.device))
        #             cur_mask.append(torch.ones(img_feat[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(torch.full((img_feat[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             cur_embeds.append(img_end_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(img_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((img_end_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #         else:
        #             cur_embeds.append(img_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(img_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((img_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             cur_embeds.append(no_img_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(no_img_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((no_img_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             cur_embeds.append(img_end_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(img_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((img_end_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #         # 判断是否全为0
        #         audio_all_zero = audio_tag[i][ui] == 0
        #         if audio_all_zero == False:
        #             aud_np = torch.from_numpy(np.load('data/mod_feature/aud/' + audio_feature[i][ui] + '.npy')).to(
        #                 device)
        #
        #             aud_feat = self.audio_projector(aud_np.to(torch.bfloat16))
        #
        #             cur_embeds.append(audio_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(audio_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((audio_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             cur_embeds.append(aud_feat[0].to(emb.device))
        #             cur_mask.append(torch.ones(aud_feat[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(torch.full((aud_feat[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             cur_embeds.append(audio_end_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(audio_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((audio_end_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #         else:
        #             cur_embeds.append(audio_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(audio_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((audio_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             cur_embeds.append(no_aud_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(no_aud_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((no_aud_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             cur_embeds.append(audio_end_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(audio_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((audio_end_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             # 判断是否全为0
        #         vid_all_zero = vid_tag[i][ui] == 0
        #         if vid_all_zero == False:
        #             vid_np = torch.from_numpy(np.load('data/mod_feature/vid/' + vid_feature[i][ui] + '.npy')).unsqueeze(
        #                 0).to(
        #                 device)
        #
        #             vid_feat = self.vid_projector(vid_np.to(torch.bfloat16))
        #
        #             cur_embeds.append(vid_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(vid_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((vid_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             cur_embeds.append(vid_feat[0].to(emb.device))
        #             cur_mask.append(torch.ones(vid_feat[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((vid_feat[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             cur_embeds.append(vid_end_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(vid_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(torch.full((vid_end_token_embed[0].size(0),), -100, dtype=labels.dtype,
        #                                         device=labels.device))
        #
        #         else:
        #             cur_embeds.append(vid_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(vid_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(
        #                 torch.full((vid_token_embed[0].size(0),), -100, dtype=labels.dtype, device=labels.device))
        #
        #             cur_embeds.append(no_vid_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(no_vid_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(torch.full((no_vid_token_embed[0].size(0),), -100, dtype=labels.dtype,
        #                                         device=labels.device))
        #
        #             cur_embeds.append(vid_end_token_embed[0].to(emb.device))
        #             cur_mask.append(torch.ones(vid_end_token_embed[0].size(0), dtype=emb.dtype).to(emb.device))
        #             cur_label.append(torch.full((vid_end_token_embed[0].size(0),), -100, dtype=labels.dtype,
        #                                         device=labels.device))
        #
        #     cur_embeds.append(emb[u_ranges[-1][1]:])
        #     cur_mask.append(torch.ones(label.size(0), dtype=emb.dtype).to(emb.device))
        #     cur_label.append(label)
        #     # cur_embeds = [e.unsqueeze(0) if e.dim() == 1 else e for e in cur_embeds]
        #
        #     # 拼接 utterance+image
        #     new_embeddings.append(torch.cat(cur_embeds, dim=0))  # (L + num_utterance, H)
        #     new_attn_masks.append(torch.cat(cur_mask, dim=0))
        #     new_labels.append(torch.cat(cur_label, dim=0))

        # 对所有 batch 内 pad 到相同长度
        # gcn_embeds = pad_sequence(new_embeddings, batch_first=True, padding_value=self.pad_token_id).to(
        #     torch.bfloat16)  # (B, L', H)
        # if attention_mask is not None:
        #     attention_mask = pad_sequence(new_attn_masks, batch_first=True, padding_value=0).to(torch.bfloat16)
        # if labels is not None:
        #     labels = pad_sequence(new_labels, batch_first=True, padding_value=-100).to(torch.long)

        gcn_embeds = pad_sequence(gcn_embeds, batch_first=True, padding_value=self.pad_token_id).to(
            torch.bfloat16)  # (B, L', H)
        if attention_mask is not None:
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0).to(torch.bfloat16)
        if labels is not None:
            labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(torch.long)

        outputs = self.text_model(
            inputs_embeds=gcn_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs
