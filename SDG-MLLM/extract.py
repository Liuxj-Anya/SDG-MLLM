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
    edge_list = []
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                m1_start, m1_end = cluster[i]
                m2_start, m2_end = cluster[j]
                for a in range(m1_start, m1_end + 1):
                    for b in range(m2_start, m2_end + 1):
                        if a in token_mapping and b in token_mapping:
                            ta, tb = token_mapping[a], token_mapping[b]
                            edge_list.extend([(ta, tb), (tb, ta)])
    return torch.tensor(edge_list, dtype=torch.long).T if edge_list else torch.empty((2, 0), dtype=torch.long)

# ===== 情感传播边 =====
def extract_emotion_edges(doc_list, offset_list):
    edge_list = []
    for doc_idx, doc in enumerate(doc_list):
        emo_indices = [i for i, tok in enumerate(doc) if is_emotion_word(tok.text, tok.pos_)]
        ent_indices = [i for i, tok in enumerate(doc) if tok.pos_ in ("NOUN", "PROPN", "PRON")]
        offset = offset_list[doc_idx]
        for ei in emo_indices:
            for ej in ent_indices:
                edge_list.extend([(ei + offset, ej + offset), (ej + offset, ei + offset)])
    return torch.tensor(edge_list, dtype=torch.long).T if edge_list else torch.empty((2, 0), dtype=torch.long)

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
                    current_arg = tag[2:]
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
            for _, start_idx, end_idx in arg_spans:
                for arg_idx in range(start_idx, end_idx + 1):
                    edge_list.extend([
                        (pred_pos + offset, arg_idx + offset),
                        (arg_idx + offset, pred_pos + offset)
                    ])
    return torch.tensor(edge_list, dtype=torch.long).T if edge_list else torch.empty((2, 0), dtype=torch.long)

# ===== 轮次边（轮中代表token连接）=====
def extract_round_edges_central(doc_list, offset_list, speaker_list):
    from collections import defaultdict
    edge_list = []
    round_buckets = defaultdict(list)
    last_speaker = None
    round_id = 0
    for i, spk in enumerate(speaker_list):
        if spk != last_speaker:
            round_id += 1
        round_buckets[round_id].append(i)
        last_speaker = spk
    sent_representatives = []
    for doc, offset in zip(doc_list, offset_list):
        centers = [tok.i + offset for tok in doc if tok.dep_ in ("nsubj", "nsubjpass", "ROOT")]
        if not centers and doc:
            centers.append(doc[0].i + offset)
        sent_representatives.append(centers)

    # 连接同轮内句子代表token（保持原有逻辑）
    for sent_ids in round_buckets.values():
        for i in range(len(sent_ids)):
            for j in range(i + 1, len(sent_ids)):
                reps_i = sent_representatives[sent_ids[i]]
                reps_j = sent_representatives[sent_ids[j]]
                for ti in reps_i:
                    for tj in reps_j:
                        edge_list.append((ti, tj))
                        edge_list.append((tj, ti))

    # 连接相邻轮次之间的句子代表token
    round_ids = sorted(round_buckets.keys())
    for idx in range(len(round_ids) - 1):
        cur_round = round_buckets[round_ids[idx]]
        next_round = round_buckets[round_ids[idx + 1]]
        for i in cur_round:
            for j in next_round:
                reps_i = sent_representatives[i]
                reps_j = sent_representatives[j]
                for ti in reps_i:
                    for tj in reps_j:
                        edge_list.append((ti, tj))
                        edge_list.append((tj, ti))

    # print(f"[extract_round_edges_central] round_buckets: {dict(round_buckets)}")
    # print(f"[extract_round_edges_central] sent_representatives: {sent_representatives}")
    # print(f"[extract_round_edges_central] generated {len(edge_list)} edges")

    return torch.tensor(edge_list, dtype=torch.long).T if edge_list else torch.empty((2, 0), dtype=torch.long)

# ===== 图构建主函数 =====
def build_word_level_dialog_graph(dialog,
                                  use_coref=True, use_emotion=True, use_srl=True,
                                  use_speaker=True, use_turn=True,doc_id = None):
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

        # 依存边（句内）
        dep_edges = [(tok.i + start, tok.head.i + start) for tok in doc if tok.head.i != tok.i]
        dep_edges += [(b, a) for (a, b) in dep_edges]
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
        add_edges_with_type(extract_round_edges_central(doc_list, offset_list, speaker_list), 5)

    edge_index = torch.cat(total_edge_list, dim=1) if total_edge_list else torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.cat(total_edge_types) if total_edge_types else torch.empty((0,), dtype=torch.long)
    data_to_save = {
        "total_words": total_words,  # list[str]
        "edge_index": edge_index,  # torch.Tensor (2, N)
        "edge_type": edge_type,  # torch.Tensor (N,)
        "utterance_ranges": utterance_ranges  # list[tuple(int, int)]
    }

    #torch.save(data_to_save, "/data2/liuxj/1-mcabsa/data/edge/dialog_graph"+doc_id+".pt")
    return total_words, edge_index, edge_type, utterance_ranges

def data_process(data):
        # 构建 id 到 name 的映射字典
        doc_id = data['doc_id']
        id2name = {s['id']: s['name'] for s in data['speakers']}
        # 替换 speaker id 为 name
        dialogue_list = []
        print(doc_id)

        for d in data['dialogue']:
            d['speaker'] = id2name[d['speaker']]
            d['utterance'] = re.sub(r'</?[^<>]+>', '', d['utterance']).strip()
            utterance = d['speaker'] + ':' + d['utterance']
            dialogue_list.append(utterance)
        build_word_level_dialog_graph(dialogue_list,doc_id=doc_id)


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
    plt.savefig("dialogue.png")

# ===== 示例对话调用 =====
if __name__ == "__main__":
    # dialog = [
    #     "Nina:I've been thinking about upgrading my personal computer, especially since the processing speed these days seems incredibly slow.",
    #     "Alex:I totally agree! The performance of my old machine is frustrating. It's like waiting for paint to dry when running multiple applications.",]
    # words, edge_index, edge_type, _ = build_word_level_dialog_graph(dialog)
    # #words, edge_index, edge_type, _ = build_word_level_dialog_graph(dialog)
    #
    # print(edge_index.shape)
    # # （可选）边类型映射
    # edge_type_names = {
    #     0: 'dep',  # 依存
    #     1: 'coref',  # 共指
    #     2: 'srl',  # 说话人边
    #     3: 'emo',  # 情绪传播
    #     4: 'speaker',  # SRL边
    #     5: 'turn'  # 轮次
    # }
    #
    # visualize_word_graph(words, edge_index, edge_type, edge_type_names=edge_type_names)

    # with open('/data2/liuxj/1-mcabsa/data/fsex1.json', 'r', encoding='utf-8') as file:
    #     samples = json.load(file)
    #
    # for data in samples:
    #     data_process(data)

    # graph_data = torch.load("/data2/liuxj/1-mcabsa/data/edge/dialog_graph" + '4898' + ".pt")
    # print(graph_data['edge_index'].shape)
    dialog = [
        "Claire: I've just come back from a luxury cruise through the Mediterranean. The experience was indescribably beautiful, especially as we sailed past the iconic coastlines.",
        "Michael: You know, after hearing your insights, I think my view is shifting. Maybe I should consider it"
    ]
    words, edge_index, edge_type, _ = build_word_level_dialog_graph(dialog)
    visualize_word_graph(words, edge_index, edge_type)


