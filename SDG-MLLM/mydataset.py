import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
import random
import re
from dialog_graph_builder1 import *

class MyDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):

        self.tokenizer = tokenizer
        # 打开并读取JSON文件
        with open(data_path, 'r', encoding='utf-8') as file:
            self.samples = json.load(file)

    def __len__(self):
        return len(self.samples)*2

    def data_process(self,data,tokenizer,task_idx):

        doc_id=data['doc_id']
        mod_list = []

        for d in data['dialogue']:
            if 'modality' in d:
                if d['modality'] == 'None':
                    mod_list.append(d['modality'])
                else:
                    mod_list.append((d['modality']['type'], d['modality']['caption'], d['modality']['id']))
            else:
                mod_list.append('None')

        if task_idx == 0:
            # answer = [
            #     (
            #         'holder:'+str(item['holder']['value']),
            #         'target:'+str(item['target']['value']),
            #         'aspect:'+str(item['aspect']['value']),
            #         'opinion:'+str(item['opinion']['value']),
            #         'sentiment:'+str(item['sentiment']),
            #         'rationale:'+str(item['rationale']['value'])
            #     )
            answer = [
                (
                    str(item['holder']['value']),
                    str(item['target']['value']),
                    str(item['aspect']['value']),
                    str(item['opinion']['value']),
                    str(item['sentiment']),
                    str(item['rationale']['value'])
                )
                for item in data['hexatuple']
            ]
            input_text = "You are a sentiment reasoning expert. Extract all sentiment sextuples in the form: (holder, target, aspect, opinion, sentiment, rationale) , sentiment ∈ {positive, negative, neutral}, h/t/a/o/r can be explicit in text or inferred from context or non-text modalities (e.g., image, audio, video). Dialogue: "
        else:
            if 'sentiment flip' in data:
                answer = data["sentiment flip"]
                if isinstance(answer,list):
                    if len(answer)==0:
                        answer = "None"
                    else:
                        answer = answer[0]
            elif 'sentiment_flip' in data:
                answer = data["sentiment_flip"]
                if isinstance(answer, list):
                    if len(answer) == 0:
                        answer = "None"
                    else:
                        answer = answer[0]
            else:
                answer = "None"
            input_text = "You are a sentiment flipping analyst. From the dialogue, extract all flipping sextuples: (holder, target, aspect, initial_sentiment, flipped_sentiment, trigger), initial_sentiment, flipped_sentiment ∈ {positive, negative, neutral}, trigger ∈ {introduction of new information, logical argumentation, participant feedback and interaction, personal experience and self-reflection}, All flips must come from the same speaker, on the same target and aspect. Dialogue:"

        graph_data = torch.load("/data2/liuxj/1-mcabsa/data/edge/dialog_graph"+doc_id+".pt")
        graph_input = map_word_edges_to_subwords(graph_data,tokenizer)

        image_feature=[]
        audio_feature=[]
        vid_feature = []

        image_tag=[]
        audio_tag=[]
        vid_tag = []
        # shared_image_placeholder = np.zeros((576, 1536))
        # shared_audio_placeholder = np.zeros((273, 768))
        for mod in mod_list:
            if mod=='None':
                # image_feature.append(shared_image_placeholder)
                # audio_feature.append(shared_audio_placeholder)
                image_tag.append(0)
                audio_tag.append(0)
                vid_tag.append(0)
                image_feature.append('None')
                audio_feature.append('None')
                vid_feature.append('None')

            elif mod[0]=='aud':
                #aud_fea = np.squeeze(np.load('data/mod_feature/aud/'+mod[2]+'.npy'))
                audio_feature.append(mod[2])
                image_feature.append('None')
                vid_feature.append('None')
                image_tag.append(0)
                audio_tag.append(1)
                vid_tag.append(0)

            elif mod[0]=='img':
                #image_fea = np.squeeze(np.load('data/mod_feature/image/'+mod[2]+'.npy'))
                image_feature.append(mod[2])
                audio_feature.append('None')
                vid_feature.append('None')
                image_tag.append(1)
                audio_tag.append(0)
                vid_tag.append(0)

            elif mod[0]=='vid':
                #image_fea = np.squeeze(np.load('data/mod_feature/image/'+mod[2]+'.npy'))
                vid_feature.append(mod[2])
                image_feature.append('None')
                audio_feature.append('None')
                image_tag.append(0)
                audio_tag.append(0)
                vid_tag.append(1)

        #return torch.from_numpy(np.array(image_feature)),torch.from_numpy(np.array(audio_feature)),input_text,graph_input,str(answer)
        return image_feature, audio_feature,vid_feature,image_tag,audio_tag,vid_tag, input_text, graph_input, str(answer),task_idx

    def __getitem__(self, idx):
        sample_idx = idx // 2  # 原始数据 index
        task_idx = idx % 2  # 0 表示 task1，1 表示 task2
        image_feature,audio_feature,vid_feature,image_tag,audio_tag,vid_tag,input_text,graph_input,target_text,task_idx = self.data_process(self.samples[sample_idx],self.tokenizer,task_idx)

        input_enc = self.tokenizer(
            f"<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors="pt"
            )
        
        response = self.tokenizer(f"{target_text+self.tokenizer.eos_token}",
                                  add_special_tokens=False,
                                  padding=False,
                                  truncation=True,
                                  return_tensors="pt"
                                  )

        # 拼接 input_ids, attention_mask, labels
        input_ids = torch.cat([input_enc["input_ids"],graph_input[0].unsqueeze(0)], dim=1).squeeze(0)

        result={
            "input_ids": [input_ids,response["input_ids"].squeeze(0)],
            #"response_ids": response["input_ids"].squeeze(0),  # 单独的 response
            "edge_index": graph_input[1],
            "edge_type": graph_input[2],
            #"graph_token_len": [],  # graph 部分 token 数（用于计算 span）
            "utterance_ranges": [graph_input[-1],graph_input[0].shape[0],task_idx],
            "image_feature": image_feature,
            "audio_feature": audio_feature,
            "vid_feature": vid_feature,
            "image_tag": image_tag,
            "audio_tag": audio_tag,
            "vid_tag": vid_tag,
        }
        return result


class SextupleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):

        self.tokenizer = tokenizer
        # 打开并读取JSON文件
        with open(data_path, 'r', encoding='utf-8') as file:
            self.samples = json.load(file)

    def __len__(self):
        return len(self.samples)

    def data_process(self, data, tokenizer, task_idx):

        doc_id = data['doc_id']

        # 构建 id 到 name 的映射字典
        id2name = {s['id']: s['name'] for s in data['speakers']}
        # 替换 speaker id 为 name
        dialogue_list = []
        mod_list = []

        for d in data['dialogue']:

            if 'modality' in d:
                if d['modality'] == 'None':
                    mod_list.append(d['modality'])
                else:
                    mod_list.append((d['modality']['type'], d['modality']['caption'], d['modality']['id']))
            else:
                mod_list.append('None')

        answer = [
                (
                    item['holder']['value'],
                    item['target']['value'],
                    item['aspect']['value'],
                    item['opinion']['value'],
                    item['sentiment'],
                    item['rationale']['value']
                )
                for item in data['hexatuple']
            ]
        # input_text = "You are a sentiment reasoning expert for multimodal dialogues. Given a dialogue composed of multiple utterances, your task is to extract all sentiment sextuples in the format: [(holder, target, aspect, opinion, sentiment, rationale)], Where: holder (h): the person expressing the sentiment, target (t): the entity being discussed, aspect (a): the attribute or part of the target, opinion (o): the opinionated expression, sentiment (s): one of positive, negative, or neutral, rationale (r): the reason supporting the sentiment. Each utterance may contain text and/or non-text modalities (e.g., images, audio, or video). Important: The elements h, t, a, o, and r can be either continuous text spans explicitly mentioned in the utterances, or implicitly inferred from dialogue context or non-text modalities. The sentiment (s) must be selected from the three categories: positive, negative, or neutral. Dialogue: "
        input_text = "You are a sentiment reasoning expert. Extract all sentiment sextuples in the form: (holder, target, aspect, opinion, sentiment, rationale) , sentiment ∈ {positive, negative, neutral}, h/t/a/o/r can be explicit in text or inferred from context or non-text modalities (e.g., image, audio, video). Dialogue: "

        graph_data = torch.load("/data2/liuxj/1-mcabsa/data/edge/dialog_graph" + doc_id + ".pt")
        graph_input = map_word_edges_to_subwords(graph_data, tokenizer)

        image_feature = []
        audio_feature = []
        vid_feature = []

        image_tag = []
        audio_tag = []
        vid_tag = []
        # shared_image_placeholder = np.zeros((576, 1536))
        # shared_audio_placeholder = np.zeros((273, 768))
        for mod in mod_list:
            if mod == 'None':
                # image_feature.append(shared_image_placeholder)
                # audio_feature.append(shared_audio_placeholder)
                image_tag.append(0)
                audio_tag.append(0)
                vid_tag.append(0)
                image_feature.append('None')
                audio_feature.append('None')
                vid_feature.append('None')

            elif mod[0] == 'aud':
                # aud_fea = np.squeeze(np.load('data/mod_feature/aud/'+mod[2]+'.npy'))
                audio_feature.append(mod[2])
                image_feature.append('None')
                vid_feature.append('None')
                image_tag.append(0)
                audio_tag.append(1)
                vid_tag.append(0)

            elif mod[0] == 'img':
                # image_fea = np.squeeze(np.load('data/mod_feature/image/'+mod[2]+'.npy'))
                image_feature.append(mod[2])
                audio_feature.append('None')
                vid_feature.append('None')
                image_tag.append(1)
                audio_tag.append(0)
                vid_tag.append(0)

            elif mod[0] == 'vid':
                # image_fea = np.squeeze(np.load('data/mod_feature/image/'+mod[2]+'.npy'))
                vid_feature.append(mod[2])
                image_feature.append('None')
                audio_feature.append('None')
                image_tag.append(0)
                audio_tag.append(0)
                vid_tag.append(1)

        # return torch.from_numpy(np.array(image_feature)),torch.from_numpy(np.array(audio_feature)),input_text,graph_input,str(answer)
        return (image_feature, audio_feature, vid_feature, image_tag, audio_tag, vid_tag, input_text, graph_input,
                str(answer), task_idx)

    def __getitem__(self, idx):
        sample_idx = idx  # 原始数据 index
        task_idx = 0
        image_feature, audio_feature, vid_feature, image_tag, audio_tag, vid_tag, input_text, graph_input, target_text, task_idx = self.data_process(
            self.samples[sample_idx], self.tokenizer, task_idx)

        input_enc = self.tokenizer(
            f"<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        response = self.tokenizer(f"{target_text + self.tokenizer.eos_token}",
                                  add_special_tokens=False,
                                  padding=False,
                                  truncation=True,
                                  return_tensors="pt"
                                  )

        # 拼接 input_ids, attention_mask, labels
        input_ids = torch.cat([input_enc["input_ids"], graph_input[0].unsqueeze(0)], dim=1).squeeze(0)

        result = {
            "input_ids": [input_ids, response["input_ids"].squeeze(0)],
            # "response_ids": response["input_ids"].squeeze(0),  # 单独的 response
            "edge_index": graph_input[1],
            "edge_type": graph_input[2],
            # "graph_token_len": [],  # graph 部分 token 数（用于计算 span）
            "utterance_ranges": [graph_input[-1], graph_input[0].shape[0], task_idx],
            "image_feature": image_feature,
            "audio_feature": audio_feature,
            "vid_feature": vid_feature,
            "image_tag": image_tag,
            "audio_tag": audio_tag,
            "vid_tag": vid_tag,
        }
        return result


class MultiDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):
        self.tokenizer = tokenizer
        # 打开并读取JSON文件
        with open(data_path, 'r', encoding='utf-8') as file:
            self.samples = json.load(file)

    def __len__(self):
        return len(self.samples) * 8

    def data_process(self, data, tokenizer, task_idx):
        doc_id = data['doc_id']
        mod_list = []
        for d in data['dialogue']:
            if 'modality' in d:
                if d['modality'] == 'None':
                    mod_list.append(d['modality'])
                else:
                    mod_list.append((d['modality']['type'], d['modality']['caption'], d['modality']['id']))
            else:
                mod_list.append('None')

        if task_idx == 0:
            answer = [
                (
                    'holder:' + str(item['holder']['value']),
                    'target:' + str(item['target']['value']),
                    'aspect:' + str(item['aspect']['value']),
                    'opinion:' + str(item['opinion']['value']),
                    'sentiment:' + str(item['sentiment']),
                    'rationale:' + str(item['rationale']['value'])
                )
                for item in data['hexatuple']
            ]
            input_text = "You are a sentiment reasoning expert. Extract all sentiment sextuples in the form: (holder, target, aspect, opinion, sentiment, rationale) , sentiment ∈ {positive, negative, neutral}, h/t/a/o/r can be explicit in text or inferred from context or non-text modalities (e.g., image, audio, video). Dialogue: "

        elif task_idx == 1:
            answer = [
                (
                    'holder:' + str(item['holder']['value']),
                    'target:' + str(item['target']['value']),
                    'aspect:' + str(item['aspect']['value']),
                    'opinion:' + str(item['opinion']['value']),
                    'sentiment:' + str(item['sentiment']),
                    'rationale:' + str(item['rationale']['value'])
                )
                for item in data['hexatuple']
            ]
            input_text = "You are a sentiment reasoning expert. Extract all sentiment sextuples in the form: (holder, target, aspect, opinion, sentiment, rationale) , sentiment ∈ {positive, negative, neutral}, h/t/a/o/r can be explicit in text or inferred from context or non-text modalities (e.g., image, audio, video). Dialogue: "

        elif task_idx == 2:
            answer = [
                (
                    'holder:' + str(item['holder']['value']),
                    'target:' + str(item['target']['value']),
                    'aspect:' + str(item['aspect']['value']),
                    'opinion:' + str(item['opinion']['value']),
                    'sentiment:' + str(item['sentiment']),
                    'rationale:' + str(item['rationale']['value'])
                )
                for item in data['hexatuple']
            ]
            input_text = "You are a sentiment reasoning expert. Extract all sentiment sextuples in the form: (holder, target, aspect, opinion, sentiment, rationale) , sentiment ∈ {positive, negative, neutral}, h/t/a/o/r can be explicit in text or inferred from context or non-text modalities (e.g., image, audio, video). Dialogue: "

        elif task_idx == 3:
            answer = [
                (
                    'holder:' + str(item['holder']['value']),
                    'target:' + str(item['target']['value']),
                    'aspect:' + str(item['aspect']['value']),
                    'opinion:' + str(item['opinion']['value']),
                    'sentiment:' + str(item['sentiment']),
                    'rationale:' + str(item['rationale']['value'])
                )
                for item in data['hexatuple']
            ]
            input_text = "You are a sentiment reasoning expert. Extract all sentiment sextuples in the form: (holder, target, aspect, opinion, sentiment, rationale) , sentiment ∈ {positive, negative, neutral}, h/t/a/o/r can be explicit in text or inferred from context or non-text modalities (e.g., image, audio, video). Dialogue: "

        elif task_idx == 4:
            answer = [
                (
                    'holder:' + str(item['holder']['value']),
                    'target:' + str(item['target']['value']),
                    'aspect:' + str(item['aspect']['value']),
                    'opinion:' + str(item['opinion']['value']),
                    'sentiment:' + str(item['sentiment']),
                    'rationale:' + str(item['rationale']['value'])
                )
                for item in data['hexatuple']
            ]
            input_text = "You are a sentiment reasoning expert. Extract all sentiment sextuples in the form: (holder, target, aspect, opinion, sentiment, rationale) , sentiment ∈ {positive, negative, neutral}, h/t/a/o/r can be explicit in text or inferred from context or non-text modalities (e.g., image, audio, video). Dialogue: "

        elif task_idx == 5:
            answer = [
                (
                    'holder:' + str(item['holder']['value']),
                    'target:' + str(item['target']['value']),
                    'aspect:' + str(item['aspect']['value']),
                    'opinion:' + str(item['opinion']['value']),
                    'sentiment:' + str(item['sentiment']),
                    'rationale:' + str(item['rationale']['value'])
                )
                for item in data['hexatuple']
            ]
            input_text = "You are a sentiment reasoning expert. Extract all sentiment sextuples in the form: (holder, target, aspect, opinion, sentiment, rationale) , sentiment ∈ {positive, negative, neutral}, h/t/a/o/r can be explicit in text or inferred from context or non-text modalities (e.g., image, audio, video). Dialogue: "

        elif task_idx == 6:
            answer = [
                (
                    'holder:' + str(item['holder']['value']),
                    'target:' + str(item['target']['value']),
                    'aspect:' + str(item['aspect']['value']),
                    'opinion:' + str(item['opinion']['value']),
                    'sentiment:' + str(item['sentiment']),
                    'rationale:' + str(item['rationale']['value'])
                )
                for item in data['hexatuple']
            ]
            input_text = "You are a sentiment reasoning expert. Extract all sentiment sextuples in the form: (holder, target, aspect, opinion, sentiment, rationale) , sentiment ∈ {positive, negative, neutral}, h/t/a/o/r can be explicit in text or inferred from context or non-text modalities (e.g., image, audio, video). Dialogue: "

        elif task_idx == 7:
            answer = [
                (
                    'holder:' + str(item['holder']['value']),
                    'target:' + str(item['target']['value']),
                    'aspect:' + str(item['aspect']['value']),
                    'opinion:' + str(item['opinion']['value']),
                    'sentiment:' + str(item['sentiment']),
                    'rationale:' + str(item['rationale']['value'])
                )
                for item in data['hexatuple']
            ]
            input_text = "You are a sentiment reasoning expert. Extract all sentiment sextuples in the form: (holder, target, aspect, opinion, sentiment, rationale) , sentiment ∈ {positive, negative, neutral}, h/t/a/o/r can be explicit in text or inferred from context or non-text modalities (e.g., image, audio, video). Dialogue: "
        else:
            if 'sentiment flip' in data:
                answer = data["sentiment flip"]
                if isinstance(answer, list):
                    if len(answer) == 0:
                        answer = "None"
                    else:
                        answer = answer[0]
            elif 'sentiment_flip' in data:
                answer = data["sentiment_flip"]
                if isinstance(answer, list):
                    if len(answer) == 0:
                        answer = "None"
                    else:
                        answer = answer[0]
            else:
                answer = "None"
            input_text = "You are a sentiment flipping analyst. From the dialogue, extract all flipping sextuples: (holder, target, aspect, initial_sentiment, flipped_sentiment, trigger), initial_sentiment, flipped_sentiment ∈ {positive, negative, neutral}, trigger ∈ {introduction of new information, logical argumentation, participant feedback and interaction, personal experience and self-reflection}, All flips must come from the same speaker, on the same target and aspect. Dialogue:"

        graph_data = torch.load("/data2/liuxj/1-mcabsa/data/edge/dialog_graph" + doc_id + ".pt")
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

                audio_feature.append(mod[2])
                image_feature.append('None')
                vid_feature.append('None')
                image_tag.append(0)
                audio_tag.append(1)
                vid_tag.append(0)

            elif mod[0] == 'img':

                image_feature.append(mod[2])
                audio_feature.append('None')
                vid_feature.append('None')
                image_tag.append(1)
                audio_tag.append(0)
                vid_tag.append(0)

            elif mod[0] == 'vid':
                # image_fea = np.squeeze(np.load('data/mod_feature/image/'+mod[2]+'.npy'))
                vid_feature.append(mod[2])
                image_feature.append('None')
                audio_feature.append('None')
                image_tag.append(0)
                audio_tag.append(0)
                vid_tag.append(1)

        # return torch.from_numpy(np.array(image_feature)),torch.from_numpy(np.array(audio_feature)),input_text,graph_input,str(answer)
        return image_feature, audio_feature, vid_feature, image_tag, audio_tag, vid_tag, input_text, graph_input, str(
            answer), task_idx

    def __getitem__(self, idx):
        sample_idx = idx // 8  # 原始数据 index
        task_idx = idx % 8  # 0 表示 task1，1 表示 task2
        image_feature, audio_feature, vid_feature, image_tag, audio_tag, vid_tag, input_text, graph_input, target_text, task_idx = self.data_process(
            self.samples[sample_idx], self.tokenizer, task_idx)

        input_enc = self.tokenizer(
            f"<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        response = self.tokenizer(f"{target_text + self.tokenizer.eos_token}",
                                  add_special_tokens=False,
                                  padding=False,
                                  truncation=True,
                                  return_tensors="pt"
                                  )

        # 拼接 input_ids, attention_mask, labels
        input_ids = torch.cat([input_enc["input_ids"], graph_input[0].unsqueeze(0)], dim=1).squeeze(0)

        result = {
            "input_ids": [input_ids, response["input_ids"].squeeze(0)],
            # "response_ids": response["input_ids"].squeeze(0),  # 单独的 response
            "edge_index": graph_input[1],
            "edge_type": graph_input[2],
            # "graph_token_len": [],  # graph 部分 token 数（用于计算 span）
            "utterance_ranges": [graph_input[-1], graph_input[0].shape[0], task_idx],
            "image_feature": image_feature,
            "audio_feature": audio_feature,
            "vid_feature": vid_feature,
            "image_tag": image_tag,
            "audio_tag": audio_tag,
            "vid_tag": vid_tag,
        }
        return result