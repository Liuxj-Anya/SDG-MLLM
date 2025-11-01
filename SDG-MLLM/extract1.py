import torch
import spacy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import allennlp_models.tagging
import json
import re
import matplotlib.pyplot as plt
import networkx as nx


# ===== 模型加载 =====
coref_predictor = Predictor.from_path("/data2/liuxj/1-mcabsa/coref-spanbert-large-2021.03.10.tar.gz")
srl_predictor = Predictor.from_path("/data2/liuxj/1-mcabsa/structured-prediction-srl-bert.2020.12.15.tar.gz")
nlp = spacy.load("en_core_web_trf")

# ===== 情感词典加载 =====
def load_nrc_emotion_lexicon(path):
    lexicon = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word, emotion, label = line.strip().split("\t")
            if int(label) == 1:
                lexicon[emotion].add(word.lower())
    return lexicon

nrc_path = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
lexicon_dict = load_nrc_emotion_lexicon(nrc_path)
emotion_lexicon = set()
for emo in ["joy", "sadness", "anger", "trust", "positive", "negative"]:
    emotion_lexicon |= lexicon_dict[emo]

def is_emotion_word(word, pos):
    return word.lower() in emotion_lexicon and (pos.startswith("ADJ") or pos.startswith("VERB") or pos.startswith("NOUN"))

# ===== 共指边 =====
def extract_coref_edges_allennlp_aligned(doc_list, offset_list):
    full_text = " ".join(doc.text for doc in doc_list)
    prediction = coref_predictor.predict(document=full_text)
    clusters = prediction["clusters"]
    allen_tokens = prediction["document"]
    spacy_doc = nlp(full_text)

    # AllenNLP token index -> SpaCy token index
    def build_token_mapping():
        mapping = {}
        start = 0
        for allen_idx, allen_tok in enumerate(allen_tokens):
            while start < len(spacy_doc):
                if spacy_doc[start].text == allen_tok:
                    mapping[allen_idx] = start
                    start += 1
                    break
                else:
                    start += 1
        return mapping

    token_mapping = build_token_mapping()

    # 构建每句话的 token 范围，用 offset_list 做映射
    spacy_offset_ranges = []
    idx = 0
    for doc in doc_list:
        start = idx
        end = idx + len(doc)
        spacy_offset_ranges.append((start, end))
        idx = end

    # spacy 全局 token index -> 图中全局 index（加 offset_list 的偏移）
    def spacy_idx_to_graph_idx(spacy_idx):
        for (range_start, range_end), offset in zip(spacy_offset_ranges, offset_list):
            if range_start <= spacy_idx < range_end:
                return spacy_idx - range_start + offset
        return None

    # 构建边
    edge_list = []
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                m1_start, m1_end = cluster[i]
                m2_start, m2_end = cluster[j]
                for a in range(m1_start, m1_end + 1):
                    for b in range(m2_start, m2_end + 1):
                        if a in token_mapping and b in token_mapping:
                            ta_spacy = token_mapping[a]
                            tb_spacy = token_mapping[b]
                            ta = spacy_idx_to_graph_idx(ta_spacy)
                            tb = spacy_idx_to_graph_idx(tb_spacy)
                            if ta is not None and tb is not None:
                                edge_list.append((ta, tb))
                                edge_list.append((tb, ta))  # 双向

    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).T
    else:
        return torch.empty((2, 0), dtype=torch.long)


def extract_emotion_edges(doc_list, offset_list):
    edge_list = []
    for doc_idx, doc in enumerate(doc_list):
        emo_indices = [i for i, tok in enumerate(doc) if is_emotion_word(tok.text, tok.pos_)]
        ent_indices = [i for i, tok in enumerate(doc) if tok.pos_ in ("NOUN", "PROPN")]  # 排除 PRON
        offset = offset_list[doc_idx]

        for ei in emo_indices:
            if not ent_indices:
                continue
            # 根据距离排序实体索引
            sorted_ents = sorted(ent_indices, key=lambda x: abs(x - ei))
            # 取最近3个
            top_ents = sorted_ents[:3]
            for ej in top_ents:
                edge_list.append((ei + offset, ej + offset))
                edge_list.append((ej + offset, ei + offset))  # 双向

    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).T
    else:
        return torch.empty((2, 0), dtype=torch.long)


# ===== SRL边 =====
def extract_srl_edges(doc_list, offset_list):
    edge_list = []
    for doc_idx, doc in enumerate(doc_list):
        sentence = doc.text
        offset = offset_list[doc_idx]
        srl_result = srl_predictor.predict(sentence=sentence)

        for verb in srl_result["verbs"]:
            tags = verb["tags"]
            pred_pos = next((i for i, tag in enumerate(tags) if tag == "B-V"), None)
            if pred_pos is None:
                continue

            arg_spans = []
            current_arg = None
            current_start = None

            for i, tag in enumerate(tags):
                if tag.startswith("B-ARG"):
                    if current_arg is not None:
                        arg_spans.append((current_arg, current_start, i - 1))
                    current_arg = tag[5:]  # 提取 ARG 名字 (如 0, 1, 2, M-TMP)
                    current_start = i
                elif tag.startswith("I-ARG"):
                    continue
                else:
                    if current_arg is not None:
                        arg_spans.append((current_arg, current_start, i - 1))
                        current_arg = None
                        current_start = None
            if current_arg is not None:
                arg_spans.append((current_arg, current_start, len(tags) - 1))

            # 只连主要 argument (ARG0, ARG1, ARG2)
            for arg_name, start_idx, end_idx in arg_spans:
                if arg_name not in {"0", "1", "2"}:
                    continue
                arg_token_idx = start_idx + offset
                edge_list.append((pred_pos + offset, arg_token_idx))
                edge_list.append((arg_token_idx, pred_pos + offset))

    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).T
    else:
        return torch.empty((2, 0), dtype=torch.long)


def extract_reply_edges_central(doc_list, offset_list, speaker_list, reply, max_reps_per_sent=None):
    edge_list = []

    # 每句代表 token
    sent_representatives = []
    for doc, offset in zip(doc_list, offset_list):
        centers = [tok.i + offset for tok in doc if tok.dep_ in ("nsubj", "nsubjpass", "ROOT")]
        if not centers and doc:
            centers.append(doc[0].i + offset)
        if max_reps_per_sent:
            centers = centers[:max_reps_per_sent]
        sent_representatives.append(centers)

    # 基于 reply 生成边
    for i, reply_to in enumerate(reply):
        if reply_to == -1:
            continue
        reps_i = sent_representatives[i]
        reps_j = sent_representatives[reply_to]
        for a in reps_i:
            for b in reps_j:
                edge_list.append((a, b))
                edge_list.append((b, a))  # 如果需要双向

    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).T
    else:
        return torch.empty((2, 0), dtype=torch.long)


# ===== 图构建主函数 =====
def build_word_level_dialog_graph(dialog,
                                  use_coref=True, use_emotion=True, use_srl=True,
                                  use_speaker=True, use_turn=True,doc_id = None,reply=None):
    total_words, total_edge_list, total_edge_types = [], [], []
    doc_list, offset_list, speaker_list, utterance_ranges = [], [], [], []
    offset = 0
    for utt in dialog:
        # 直接用整条utterance，不拆分speaker和内容
        doc = nlp(utt.strip())
        doc_list.append(doc)

        # 尝试简单提取说话人（冒号前前缀）作为speaker
        if ":" in utt:
            speaker = utt.split(":", 1)[0].strip()
        else:
            speaker = "unknown"
        speaker_list.append(speaker)

        start = offset
        end = offset + len(doc)
        offset_list.append(offset)
        utterance_ranges.append((start, end))
        offset = end
        total_words.extend([t.text for t in doc])

        # 依存边（句内，去除标点符号相关边）
        dep_edges = []
        for tok in doc:
            if tok.head.i != tok.i:
                # 过滤掉标点符号作为源或目标
                if tok.pos_ != "PUNCT" and tok.head.pos_ != "PUNCT":
                    src = tok.i + start
                    tgt = tok.head.i + start
                    dep_edges.append((src, tgt))
                    dep_edges.append((tgt, src))  # 如果需要双向边
        if dep_edges:
            edge_tensor = torch.tensor(dep_edges, dtype=torch.long).T
            edge_type = torch.full((edge_tensor.size(1),), 0, dtype=torch.long)
            total_edge_list.append(edge_tensor)
            total_edge_types.append(edge_type)

    def add_edges_with_type(edge_tensor, edge_type_id):
        if edge_tensor.size(1) > 0:
            edge_type = torch.full((edge_tensor.size(1),), edge_type_id, dtype=torch.long)
            total_edge_list.append(edge_tensor)
            total_edge_types.append(edge_type)

    if use_coref:
        add_edges_with_type(extract_coref_edges_allennlp_aligned(doc_list, offset_list), 1)
    if use_emotion:
        add_edges_with_type(extract_emotion_edges(doc_list, offset_list), 2)
    if use_srl:
        add_edges_with_type(extract_srl_edges(doc_list, offset_list), 3)
    if use_speaker:
        for i in range(len(speaker_list)):
            for j in range(i + 1, len(speaker_list)):
                if speaker_list[i] != speaker_list[j]:
                    a, b = utterance_ranges[i][0], utterance_ranges[j][0]
                    total_edge_list.append(torch.tensor([[a, b], [b, a]], dtype=torch.long))
                    total_edge_types.append(torch.full((2,), 4, dtype=torch.long))
    if use_turn:
        add_edges_with_type(extract_reply_edges_central(doc_list, offset_list, speaker_list, reply, max_reps_per_sent=3), 5)

    edge_index = torch.cat(total_edge_list, dim=1) if total_edge_list else torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.cat(total_edge_types) if total_edge_types else torch.empty((0,), dtype=torch.long)
    data_to_save = {
        "total_words": total_words,  # list[str]
        "edge_index": edge_index,  # torch.Tensor (2, N)
        "edge_type": edge_type,  # torch.Tensor (N,)
        "utterance_ranges": utterance_ranges  # list[tuple(int, int)]
    }

    #torch.save(data_to_save, "/data2/liuxj/1-mcabsa/test/flip_edge1/dialog_graph"+doc_id+".pt")
    return total_words, edge_index, edge_type, utterance_ranges

def data_process(data):
        # 构建 id 到 name 的映射字典
        doc_id = data['doc_id']
        id2name = {s['id']: s['name'] for s in data['speakers']}
        # 替换 speaker id 为 name
        dialogue_list = []
        reply_list=[]
        print(doc_id)
        count=-1
        for d in data['dialogue']:
            d['speaker'] = id2name[d['speaker']]
            #d['utterance'] = re.sub(r'</?[^<>]+>', '', d['utterance']).strip()
            utterance = d['speaker'] + ':' + d['utterance']
            dialogue_list.append(utterance)
            if 'reply' in d:
                reply_list.append(int(d["reply"]))
            else:
                reply_list.append(count)
                count+=1
        build_word_level_dialog_graph(dialogue_list,doc_id=doc_id,reply=reply_list)

# ===== 图可视化函数 =====
def visualize_word_graph(words, edge_index, edge_type):
    G = nx.DiGraph()
    for i, word in enumerate(words):
        G.add_node(i, label=word)

    edge_labels_map = {0: "dep", 1: "coref", 2: "emo", 3: "srl", 4: "spk", 5: "turn"}
    edge_color_map = {0: "black", 1: "blue", 2: "green", 3: "orange", 4: "purple", 5: "red"}
    edge_groups = defaultdict(list)
    edge_labels = {}

    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        t = edge_type[i].item()
        G.add_edge(u.item(), v.item(), label=edge_labels_map.get(t, "?"))
        edge_groups[t].append((u.item(), v.item()))
        edge_labels[(u.item(), v.item())] = edge_labels_map.get(t, "?")

    pos = nx.spring_layout(G, k=2.0, iterations=200)
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="lightgray")
    nx.draw_networkx_labels(G, pos, labels={i: w for i, w in enumerate(words)}, font_size=10)
    for t, edges in edge_groups.items():
        color = edge_color_map.get(t, "gray")
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=2, alpha=0.8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={e: edge_labels[e] for e in edges}, font_color=color, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("dialogue1.png")

# ===== 示例对话调用 =====
if __name__ == "__main__":

    # graph_data = torch.load("/data2/liuxj/1-mcabsa/data/edge/dialog_graph" + '4898' + ".pt")
    # print(graph_data['edge_index'].shape)

    with open('/data2/liuxj/1-mcabsa/test/MCABSA_testset/task2_input.json', 'r', encoding='utf-8') as file:
        samples = json.load(file)

    for data in samples:
        data_process(data)



