import os
from qwen3vl_hgt import *
from mydataset import *
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments,EarlyStoppingCallback
from transformers import Trainer
import numpy as np
from peft.utils import set_peft_model_state_dict, get_peft_model_state_dict
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from peft.utils import set_peft_model_state_dict, get_peft_model_state_dict
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

class DynamicEvalCallback(TrainerCallback):
    def __init__(self, schedule):
        """
        schedule: dict[int, int]，键是step阈值，值是当前阶段的评估间隔
        例如：{0:2000, 10000:1000, 20000:500}
        """
        self.schedule = schedule
        self.last_eval_step = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 获取当前阶段对应的 eval interval
        for step_threshold in sorted(self.schedule.keys(), reverse=True):
            if state.global_step >= step_threshold:
                current_interval = self.schedule[step_threshold]
                break
        else:
            current_interval = list(self.schedule.values())[0]

        if (state.global_step - self.last_eval_step) >= current_interval:
            control.should_evaluate = True
            self.last_eval_step = state.global_step

        return control

lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["embed_tokens","q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.02
            )

def load_all(save_dir, config, tokenizer=None, lora_config=None, inject_lora=False):

    # Step 1: 初始化基础模型（未注入 LoRA）
    rel_id2name = {0: 'dep', 1: 'coref', 2: 'srl', 3: 'emo', 4: 'speaker', 5: 'turn'}
    # 初始化主模型（包含 projector 和 embed）
    model = Qwen3RGCNEncoder(config=config, lora_config=None, tokenizer=tokenizer, gcn_hidden_dim=256, gcn_layers=1, rel_id2name=rel_id2name)

    # Step 2: 加载 projector 和 embed 权重
    model.image_projector.load_state_dict(torch.load(f"{save_dir}/image_projector.pt"))
    model.audio_projector.load_state_dict(torch.load(f"{save_dir}/audio_projector.pt"))
    model.vid_projector.load_state_dict(torch.load(f"{save_dir}/vid_projector.pt"))
    model.gcn.load_state_dict(torch.load(f"{save_dir}/gcn_layers.pt"))
    model.special_token_embed.load_state_dict(torch.load(f"{save_dir}/special_token_embed.pt"))

    if inject_lora and lora_config is not None:
        # Step 3a: 注入 LoRA adapter（重新初始化结构）
        model.text_model = get_peft_model(model.text_model, lora_config)

        # Step 3b: 临时加载保存的 PeftModel，用于提取 LoRA adapter 权重
        print("Loading existing LoRA adapter weights...")
        temp_peft_model = PeftModel.from_pretrained(model.text_model.model, save_dir)
        adapter_state_dict = get_peft_model_state_dict(temp_peft_model)

        # Step 3c: 注入权重到当前模型
        set_peft_model_state_dict(model.text_model, adapter_state_dict)

    else:
        # 不注入 LoRA，只是用来推理或eval的情形
        model.text_model = PeftModel.from_pretrained(model.text_model, save_dir)

    return model

def remove_self_loops(edge_index, edge_type=None):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_type is not None:
        edge_type = edge_type[mask]
        return edge_index, edge_type
    return edge_index

class MyDataCollator1:
    def __call__(self, batch):

        pad_token_id = tokenizer.pad_token_id
        input_ids_list = []
        labels_list = []
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
        task_id = torch.tensor(task_idx)
        # response_ids = [item["response_ids"] for item in batch]

        # 拼接 input_ids 和 response_ids，同时准备 labels 和 attention_mask
        for item in batch:
            prompt_graph = item["input_ids"][0]  # prompt + graph
            response = item["input_ids"][1]  # response 部分
            full_input = torch.cat([prompt_graph, response], dim=0)

            input_ids_list.append(full_input)
            attention_mask_list.append(torch.ones_like(full_input))

            # label 只监督 response 部分
            labels = torch.full_like(full_input, -100)
            labels[-response.shape[0]:] = response
            labels_list.append(labels)

            # padding
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        for i, item in enumerate(batch):
            #edge_index = item["edge_index"]
            #graph_token_len = item["utterance_ranges"][1]
            # 在 collate_fn 中
            #item["edge_index"], item["edge_type"] = remove_self_loops(item["edge_index"], item["edge_type"])
            #edge_index = item["edge_index"]

            prompt_len = item["input_ids"][0].shape[0] - item["utterance_ranges"][1]  # 推出 prompt 长度

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

        # 你原来的 custom_collate_fn 逻辑放这里
        result = {
            "input_ids": input_ids_padded,  # (B, L)
            "attention_mask": attention_mask_padded,  # (B, L)
            "labels": labels_padded,  # (B, L)
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
            "task_id": task_id
        }
        return result

#tokenizer = AutoTokenizer.from_pretrained("/data2/liuxj/1-mcabsa/best_model_best")

rel_id2name = {0: 'dep', 1: 'coref', 2: 'srl', 3: 'emo', 4: 'speaker', 5: 'turn'}
tokenizer = AutoTokenizer.from_pretrained('/data2/liuxj/1-Sentiment-mllm/model_train/Qwen/Qwen3-8B')
special_tokens_dict = {'additional_special_tokens': ['<img>', '</img>']}
tokenizer.add_special_tokens(special_tokens_dict)

# train_dataset=SextupleDataset('/data2/liuxj/1-mcabsa/data/train.json', tokenizer)
# eval_dataset=SextupleDataset('/data2/liuxj/1-mcabsa/data/val.json', tokenizer)
#
# config = TextImageConfig(text_model_path='/data2/liuxj/1-Sentiment-mllm/model_train/Qwen/Qwen3-8B')
# model = load_all("/data2/liuxj/1-mcabsa/best_model_best",
#                  config=config,
#                  tokenizer=tokenizer,
#                  lora_config=lora_config,
#                  inject_lora=True)  # 表示你要注入新的LoRA微调

train_dataset=MyDataset('/data2/liuxj/1-mcabsa/data/train_shun_8_2.json', tokenizer)
eval_dataset=MyDataset('/data2/liuxj/1-mcabsa/data/val_shun_8_2.json', tokenizer)

config = TextImageConfig(text_model_path='/data2/liuxj/1-Sentiment-mllm/model_train/Qwen/Qwen3-8B')
model = Qwen3_HGTEncoder(config,lora_config,tokenizer,gcn_layers=6,rel_id2name=rel_id2name)
#启用 gradient checkpointing，节省显存
model.text_model.gradient_checkpointing_enable()
model.config.use_cache = False  # 关闭缓存，配合 Trainer 使用时建议关闭

def save_all(model, save_dir, tokenizer=None):
    # 保存 LoRA adapter 参数
    model.text_model.save_pretrained(save_dir)

    # 保存 projector 和 special_token_embed
    torch.save(model.image_projector.state_dict(), f"{save_dir}/image_projector.pt")
    torch.save(model.audio_projector.state_dict(), f"{save_dir}/audio_projector.pt")
    torch.save(model.vid_projector.state_dict(), f"{save_dir}/vid_projector.pt")
    torch.save(model.hgt_encoder.state_dict(), f"{save_dir}/hgt_encoder_layers.pt")
    torch.save(model.special_token_embed.state_dict(), f"{save_dir}/special_token_embed.pt")
    torch.save(model.gate_layer.state_dict(), f"{save_dir}/gate_layer.pt")
    # 保存 config
    model.config.save_pretrained(save_dir)

    # 保存 tokenizer（如果传入）
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 保存 tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # 保存模型权重（包括 LoRA 和额外组件）
        save_all(self.model, output_dir, tokenizer=self.tokenizer)

training_args = TrainingArguments(
    output_dir="./checkpoints4",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=20,
    eval_strategy="steps",
    eval_steps=150,
    save_strategy="steps",
    save_steps=150,
    save_total_limit=3,
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    deepspeed="deepspeed_config.json",
    report_to="none",
    #label_smoothing_factor=0.2,

    #EarlyStopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# 定义动态 eval 步数计划
eval_schedule = {
    0: 150,       # 前期每2000步
    10000: 150,   # 1万步后每1000步
    20000: 150     # 2万步后每500步
}

# 实例化 Trainer，添加 EarlyStoppingCallback
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    #data_collator=custom_collate_fn,
    data_collator=MyDataCollator1(),
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]  #添加 EarlyStopping
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=8),
        DynamicEvalCallback(schedule=eval_schedule)  # ✅ 添加动态评估Callback
    ]
)
trainer.train()
trainer.save_model("best_model")          # 将当前 model（已是最佳）保存到 best_model/
tokenizer.save_pretrained("best_model")   # 保存 tokenizer（包含特定 vocab，如 <img>、</img>）

# 评估并提取 eval_loss
eval_metrics = trainer.evaluate()
eval_loss = eval_metrics.get("eval_loss", None)

# 保存 eval_loss 到 best_model/metrics.json
if eval_loss is not None:
    import json
    with open("best_model/metrics.json", "w") as f:
        json.dump({"eval_loss": eval_loss}, f, indent=2)