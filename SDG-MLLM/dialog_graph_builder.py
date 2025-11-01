import torch
import spacy
import coreferee
import en_core_web_sm

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("coreferee")

# 加载情感词典
def load_nrc_emotion_lexicon(path):
    from collections import defaultdict
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

# ===== 工具函数 =====
def compute_token_offsets(token_ranges_all, start=1):
    offset_list = [start]
    for token_ranges in token_ranges_all[:-1]:
        total = sum(e - s for s, e in token_ranges)
        offset_list.append(offset_list[-1] + total)
    return offset_list

# ===== 共指边 =====
def extract_coref_edges(doc_list, token_ranges_all):
    full_text = " ".join(doc.text for doc in doc_list)
    coref_doc = nlp(full_text)

    edge_list = []

    # 遍历每个共指链
    for chain in coref_doc._.coref_chains:
        mentions = chain.mentions  # 注意：mentions 是 Mention 对象
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                m1 = mentions[i]
                m2 = mentions[j]
                try:
                    for a in range(m1.start, m1.end):
                        for b in range(m2.start, m2.end):
                            edge_list.append((a, b))
                            edge_list.append((b, a))
                except Exception as e:
                    continue

    return torch.tensor(edge_list, dtype=torch.long).T if edge_list else torch.empty((2, 0), dtype=torch.long)

# ===== 情感传播边 =====
def extract_emotion_edges(doc_list, token_ranges_all):
    offset_list = compute_token_offsets(token_ranges_all)
    edge_list = []

    for doc_idx, doc in enumerate(doc_list):
        token_ranges = token_ranges_all[doc_idx]
        emo_indices = [i for i, tok in enumerate(doc) if is_emotion_word(tok.text, tok.pos_)]
        ent_indices = [i for i, tok in enumerate(doc) if tok.pos_ in ("NOUN", "PROPN", "PRON")]

        emo_offset = offset_list[doc_idx]
        for ei in emo_indices:
            for ej in ent_indices:
                for a in range(*token_ranges[ei]):
                    for b in range(*token_ranges[ej]):
                        edge_list.extend([(a + emo_offset, b + emo_offset), (b + emo_offset, a + emo_offset)])

    return torch.tensor(edge_list, dtype=torch.long).T if edge_list else torch.empty((2, 0), dtype=torch.long)

# ===== SRL边 =====
def extract_srl_edges(doc_list, token_ranges_all):
    edge_list = []
    offset_list = compute_token_offsets(token_ranges_all, start=0)

    for doc_idx, doc in enumerate(doc_list):
        sentence = doc.text
        token_ranges = token_ranges_all[doc_idx]
        offset = offset_list[doc_idx]

        parsed = nlp(sentence)

        for token in parsed:
            # 将谓词视为动词
            if token.pos_ == "VERB":
                pred_idx = token.i
                pred_range = token_ranges[pred_idx] if pred_idx < len(token_ranges) else None
                if pred_range is None:
                    continue

                for child in token.children:
                    # 如果是主语/宾语/间宾，视为论元
                    if child.dep_ in {"nsubj", "dobj", "iobj", "attr", "obl"}:
                        arg_idx = child.i
                        arg_range = token_ranges[arg_idx] if arg_idx < len(token_ranges) else None
                        if arg_range is None:
                            continue

                        for a in range(*pred_range):
                            for b in range(*arg_range):
                                edge_list.extend([(a + offset, b + offset), (b + offset, a + offset)])

    return torch.tensor(edge_list, dtype=torch.long).T if edge_list else torch.empty((2, 0), dtype=torch.long)
# ===== 主图构建函数 =====
def build_enhanced_dialog_graph(dialog, tokenizer,
                                use_coref=True, use_emotion=True, use_srl=True,
                                use_speaker=True, use_turn=True):

    total_input_ids = []
    total_edge_list = []
    total_edge_types = []
    token_ranges_all = []
    doc_list = []
    speaker_list = []
    offset = 1

    for utt in dialog:
        if ":" in utt:
            speaker, content = utt.split(":", 1)
        else:
            speaker, content = "unknown", utt
        speaker_list.append(speaker.strip())

        doc = nlp(content.strip())
        words = [t.text for t in doc]
        word2tokens = [tokenizer(w, add_special_tokens=False)["input_ids"] for w in words]

        token_ranges = []
        pos = offset
        for tids in word2tokens:
            token_ranges.append((pos, pos + len(tids)))
            pos += len(tids)
        # print('xxxx')
        # print(word2tokens)
        # print(token_ranges)

        # ===== 依存边 =====
        dep_edges = []
        for i, tok in enumerate(doc):
            if tok.head.i == i:
                continue
            for a in range(*token_ranges[i]):
                for b in range(*token_ranges[tok.head.i]):
                    dep_edges.extend([(a, b), (b, a)])

        if dep_edges:
            edge_tensor = torch.tensor(dep_edges, dtype=torch.long).T
            edge_type = torch.full((edge_tensor.size(1),), 0, dtype=torch.long)
            total_edge_list.append(edge_tensor)
            total_edge_types.append(edge_type)

        doc_list.append(doc)
        token_ranges_all.append(token_ranges)
        total_input_ids.extend([tid for tids in word2tokens for tid in tids])
        offset = pos

    total_input_ids.append(tokenizer.eos_token_id)

    # ===== 其他类型边 =====
    def add_edges_with_type(edge_tensor, edge_type_id):
        if edge_tensor.size(1) > 0:
            edge_type = torch.full((edge_tensor.size(1),), edge_type_id, dtype=torch.long)
            total_edge_list.append(edge_tensor)
            total_edge_types.append(edge_type)

    if use_coref:
        add_edges_with_type(extract_coref_edges(doc_list, token_ranges_all), 1)
    if use_emotion:
        add_edges_with_type(extract_emotion_edges(doc_list, token_ranges_all), 2)
    if use_srl:
        add_edges_with_type(extract_srl_edges(doc_list, token_ranges_all), 3)

    # ===== 说话人边 =====
    if use_speaker and speaker_list is not None:
        n = len(speaker_list)
        for i in range(n):
            for j in range(i + 1, n):
                if speaker_list[i] != speaker_list[j]:
                    # 第一句话的第一个token
                    start_i = token_ranges_all[i][0][0]
                    start_j = token_ranges_all[j][0][0]
                    # 双向连边
                    total_edge_list.append(torch.tensor([[start_i, start_j], [start_j, start_i]], dtype=torch.long))
                    total_edge_types.append(torch.full((2,), 4, dtype=torch.long))  # 4 是说话人边类型

    # ===== 轮次边 =====
    if use_turn:
        for i in range(len(token_ranges_all) - 1):
            # 获取当前轮次和下一轮的首 token 位置
            start_i = token_ranges_all[i][0][0]
            start_j = token_ranges_all[i + 1][0][0]

            # 连边（双向）
            total_edge_list.append(torch.tensor([[start_i, start_j], [start_j, start_i]], dtype=torch.long))
            total_edge_types.append(torch.full((2,), 5, dtype=torch.long))  # 5 是轮次边类型

    edge_index = torch.cat(total_edge_list, dim=1) if total_edge_list else torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.cat(total_edge_types) if total_edge_types else torch.empty((0,), dtype=torch.long)

    # print(f"Dependency edges: {edge_type.tolist().count(0)}")
    # print(f"Coreference edges: {edge_type.tolist().count(1)}")
    # print(f"Emotion edges: {edge_type.tolist().count(2)}")
    # print(f"SRL edges: {edge_type.tolist().count(3)}")
    # print(f"Speaker edges: {edge_type.tolist().count(4)}")
    # print(f"Turn edges: {edge_type.tolist().count(5)}")

    utterance_ranges = [
        (token_ranges[0][0], token_ranges[-1][1])
        for token_ranges in token_ranges_all
    ]

    return (
        torch.tensor(total_input_ids, dtype=torch.long),
        edge_index,
        edge_type,
        token_ranges_all,
        utterance_ranges
    )
