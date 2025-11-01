import torch
import numpy as np
from transformers import AutoTokenizer
from qwen3vl_hgt import *
from peft import LoraConfig
from peft import PeftModel
import json
from torch.utils.data import Dataset,DataLoader
import re
from dialog_graph_builder1 import *
from tqdm import tqdm

class InferDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):

        self.tokenizer = tokenizer
        # 打开并读取JSON文件
        with open(data_path, 'r', encoding='utf-8') as file:
            self.samples = json.load(file)

    def __len__(self):
        return len(self.samples) * 2

    def data_process(self, data, tokenizer, task_idx):

        doc_id = data['doc_id']

        mod_list = []

        for d in data['dialogue']:
            if 'modality' in d:
                if d['modality'] == 'None':
                    mod_list.append(d['modality'])
                elif d['modality'] == None:
                    mod_list.append('None')
                elif d['modality']['type'] == 'None':
                    mod_list.append('None')
                else:
                    mod_list.append((d['modality']['type'],d['modality']['id']))
            else:
                mod_list.append('None')

        if task_idx == 0:
            input_text = "You are a sentiment reasoning expert. Extract all sentiment sextuples in the form: (holder, target, aspect, opinion, sentiment, rationale) , sentiment ∈ {positive, negative, neutral}, h/t/a/o/r can be explicit in text or inferred from context or non-text modalities (e.g., image, audio, video). Dialogue: "
        else:
            input_text = "You are a sentiment flipping analyst. From the dialogue, extract all flipping sextuples: (holder, target, aspect, initial_sentiment, flipped_sentiment, trigger), initial_sentiment, flipped_sentiment ∈ {positive, negative, neutral}, trigger ∈ {introduction of new information, logical argumentation, participant feedback and interaction, personal experience and self-reflection}, All flips must come from the same speaker, on the same target and aspect. Dialogue:"

        graph_data = torch.load("/data2/liuxj/1-mcabsa/test/flip_edge/dialog_graph" + doc_id + ".pt")
        graph_input = map_word_edges_to_subwords(graph_data, tokenizer)

        image_feature = []
        audio_feature = []
        vid_feature = []

        image_tag = []
        audio_tag = []
        vid_tag = []

        for mod in mod_list:
            if mod == 'None':

                image_tag.append(0)
                audio_tag.append(0)
                vid_tag.append(0)
                image_feature.append('None')
                audio_feature.append('None')
                vid_feature.append('None')

            elif mod[0] == 'aud':
                # aud_fea = np.squeeze(np.load('data/mod_feature/aud/'+mod[2]+'.npy'))
                audio_feature.append(mod[1])
                image_feature.append('None')
                vid_feature.append('None')
                image_tag.append(0)
                audio_tag.append(1)
                vid_tag.append(0)

            elif mod[0] == 'img':
                # image_fea = np.squeeze(np.load('data/mod_feature/image/'+mod[2]+'.npy'))
                image_feature.append(mod[1])
                audio_feature.append('None')
                vid_feature.append('None')
                image_tag.append(1)
                audio_tag.append(0)
                vid_tag.append(0)

            elif mod[0] == 'vid':
                # image_fea = np.squeeze(np.load('data/mod_feature/image/'+mod[2]+'.npy'))
                vid_feature.append(mod[1])
                image_feature.append('None')
                audio_feature.append('None')
                image_tag.append(0)
                audio_tag.append(0)
                vid_tag.append(1)

        # return torch.from_numpy(np.array(image_feature)),torch.from_numpy(np.array(audio_feature)),input_text,graph_input,str(answer)
        return (str(doc_id),image_feature, audio_feature, vid_feature, image_tag, audio_tag, vid_tag, input_text,
                graph_input, task_idx)

    def __getitem__(self, idx):
        sample_idx = idx // 2  # 原始数据 index
        task_idx = idx % 2  # 0 表示 task1，1 表示 task2
        (doc_id,image_feature, audio_feature, vid_feature, image_tag, audio_tag, vid_tag, input_text,
         graph_input, task_idx) = self.data_process(self.samples[sample_idx], self.tokenizer, task_idx)

        input_enc = self.tokenizer(
            f"<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        # 拼接 input_ids, attention_mask, labels
        input_ids = torch.cat([input_enc["input_ids"], graph_input[0].unsqueeze(0)], dim=1).squeeze(0)

        result = {
            "input_ids": input_ids,
            "edge_index": graph_input[1],
            "edge_type": graph_input[2],
            "utterance_ranges": [graph_input[-1], graph_input[0].shape[0], task_idx,doc_id],
            "image_feature": image_feature,
            "audio_feature": audio_feature,
            "vid_feature": vid_feature,
            "image_tag": image_tag,
            "audio_tag": audio_tag,
            "vid_tag": vid_tag,
        }
        return result

class InferDataCollator:
    def __call__(self, batch):

        pad_token_id = tokenizer.pad_token_id
        input_ids_list = []
        attention_mask_list = []

        edge_index_list = []
        edge_type_list = []
        graph_spans_list = []
        utterance_ranges_list = []

        cum_token_offset = 0  # 记录每个样本的 token 偏移量（用于 graph span 定位）

        image_feature = [item["image_feature"] for item in batch]
        audio_feature = [item["audio_feature"] for item in batch]
        vid_feature = [item["vid_feature"] for item in batch]
        image_tag = [item["image_tag"] for item in batch]
        audio_tag = [item["audio_tag"] for item in batch]
        vid_tag = [item["vid_tag"] for item in batch]
        task_idx = [item["utterance_ranges"][2] for item in batch]
        doc_id = [item["utterance_ranges"][3] for item in batch]
        task_id = torch.tensor(task_idx)
        # response_ids = [item["response_ids"] for item in batch]

        # 拼接 input_ids 和 response_ids，同时准备 labels 和 attention_mask
        for item in batch:
            prompt_graph = item["input_ids"]
            full_input = torch.cat([prompt_graph], dim=0)
            input_ids_list.append(full_input)
            attention_mask_list.append(torch.ones_like(full_input))

            # padding
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        for i, item in enumerate(batch):
            prompt_len = item["input_ids"].shape[0] - item["utterance_ranges"][1]  # 推出 prompt 长度

            # 1. edge_index 不偏移
            edge_index_list.append(item["edge_index"])
            edge_type_list.append(item["edge_type"])

            # 2. graph_spans：在 full input 中的位置（需要偏移）
            graph_start = cum_token_offset + prompt_len
            graph_end = graph_start + item["utterance_ranges"][1]
            graph_spans_list.append(torch.tensor([graph_start, graph_end], dtype=torch.long))

            # 3. utterance_ranges：偏移后的在 input_ids 中的位置
            u_ranges = []
            for span in item["utterance_ranges"][0]:
                offset_span = [s + graph_start for s in span]  # graph_start = offset + prompt_len
                u_ranges.append(torch.tensor(offset_span, dtype=torch.long))
            utterance_ranges_list.append(torch.stack(u_ranges))
            cum_token_offset += input_ids_list[i].shape[0]

        edge_index_batched = torch.cat(edge_index_list, dim=1)  # 不偏移
        edge_type_batched = torch.cat(edge_type_list, dim=0)
        graph_spans_batched = torch.stack(graph_spans_list)
        utterance_ranges_batched = torch.cat(utterance_ranges_list, dim=0)

        result = {
            "input_ids": input_ids_padded,  # (B, L)
            "attention_mask": attention_mask_padded,  # (B, L)
            "edge_index": edge_index_batched,
            "edge_type": edge_type_batched,
            "graph_spans": graph_spans_batched,
            "utterance_ranges": utterance_ranges_batched,
            "image_feature": image_feature,
            "audio_feature": audio_feature,
            "vid_feature": vid_feature,
            "image_tag": image_tag,
            "audio_tag": audio_tag,
            "vid_tag": vid_tag,
            "task_id": task_id,
            "doc_id": doc_id
        }
        return result

def load_all(save_dir, config, tokenizer=None, lora_config=None):
    rel_id2name = {0: 'dep', 1: 'coref', 2: 'srl', 3: 'emo', 4: 'speaker', 5: 'turn'}
    # 初始化主模型（包含 projector 和 embed）
    model = Qwen3_HGTEncoder(config, lora_config, tokenizer, gcn_hidden_dim=256, gcn_layers=1, rel_id2name=rel_id2name)

    # 加载 projector 和 special_token_embed 权重
    model.image_projector.load_state_dict(torch.load(f"{save_dir}/image_projector.pt"))
    model.audio_projector.load_state_dict(torch.load(f"{save_dir}/audio_projector.pt"))
    model.vid_projector.load_state_dict(torch.load(f"{save_dir}/vid_projector.pt"))
    model.hgt_encoder.load_state_dict(torch.load(f"{save_dir}/hgt_encoder_layers.pt"))
    model.special_token_embed.load_state_dict(torch.load(f"{save_dir}/special_token_embed.pt"))
    model.gate_layer.load_state_dict(torch.load(f"{save_dir}/gate_layer.pt"))

    model.text_model = PeftModel.from_pretrained(model.text_model, save_dir)

    return model

# 1. 设置 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data2/liuxj/1-mcabsa/best_model_hgt5_8b")

# 3. 加载 config 和 LoRA config
config = TextImageConfig(text_model_path="/data2/liuxj/1-Sentiment-mllm/Qwen/Qwen3-8B")

# 4. 加载模型
model = load_all("/data2/liuxj/1-mcabsa/best_model_hgt5_14b", config=config, tokenizer=tokenizer, lora_config=None)
model = model.to(device)
model.eval()

eval_dataset=InferDataset('/data2/liuxj/1-mcabsa/test/MCABSA_testset/task2_input.json', tokenizer)
dataloader = DataLoader(eval_dataset, batch_size=1,collate_fn = InferDataCollator())

c=0
with torch.no_grad():
    for batch in tqdm(dataloader):
        c+=1
        if batch['task_id'].item() == 1:
            outputs = model.generate_with_image(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                edge_index=batch['edge_index'],
                edge_type=batch['edge_type'],
                graph_spans=batch["graph_spans"],
                utterance_ranges=batch["utterance_ranges"],
                image_feature=batch["image_feature"],
                audio_feature=batch["audio_feature"],
                vid_feature=batch["vid_feature"],
                image_tag=batch['image_tag'],
                audio_tag=batch['audio_tag'],
                vid_tag=batch['vid_tag'],
                task_id=batch['task_id'],
                generation_kwargs={
                    "max_new_tokens": 1000,
                    "do_sample": False,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "temperature": 0.7
                }
            )

            # 8. 解码输出
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if batch['task_id'].item() == 0:
                with open('result/Sextuple3.txt','a') as fs:
                    fs.write(batch['doc_id'][0])
                    fs.write('\t')
                    fs.write(response)
                    fs.write('\n')
            elif batch['task_id'].item() == 1:
                with open('result/Flipping3.txt', 'a') as ff:
                    ff.write(batch['doc_id'][0])
                    ff.write('\t')
                    ff.write(response)
                    ff.write('\n')

