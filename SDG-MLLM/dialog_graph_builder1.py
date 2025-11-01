import torch

def map_word_edges_to_subwords(data, tokenizer):
    total_words = data["total_words"]
    edge_index = data["edge_index"]  # (2, N)
    edge_type = data["edge_type"]    # (N,)
    utterance_ranges = data["utterance_ranges"]  # [(start_word, end_word)]

    word2subtokens = [tokenizer(w, add_special_tokens=False)["input_ids"] for w in total_words]

    # === Step 1: 生成 input_ids 和 token_ranges_all ===
    total_input_ids = []
    token_ranges = []  # 每个词在 input_ids 中的起止位置
    pos = 1  # reserve 0 for <bos> or special token
    for subtokens in word2subtokens:
        token_ranges.append((pos, pos + len(subtokens)))
        total_input_ids.extend(subtokens)
        pos += len(subtokens)
    total_input_ids.append(tokenizer.eos_token_id)

    # === Step 2: 构造 token_ranges_all (每句词的 subtoken 范围) ===
    token_ranges_all = []
    for start, end in utterance_ranges:
        token_ranges_all.append(token_ranges[start:end])

    # === Step 3: 词级边 -> 子词级边 ===
    subword_edges = []
    new_edge_types = []

    for idx in range(edge_index.size(1)):
        i, j = edge_index[0, idx].item(), edge_index[1, idx].item()
        etype = edge_type[idx].item()

        range_i = token_ranges[i]
        range_j = token_ranges[j]

        for a in range(*range_i):
            for b in range(*range_j):
                subword_edges.append((a, b))
                new_edge_types.append(etype)

    if subword_edges:
        edge_index_sub = torch.tensor(subword_edges, dtype=torch.long).T  # (2, M)
        edge_type_sub = torch.tensor(new_edge_types, dtype=torch.long)    # (M,)
    else:
        edge_index_sub = torch.empty((2, 0), dtype=torch.long)
        edge_type_sub = torch.empty((0,), dtype=torch.long)

    # === Step 4: utterance_ranges_subword (子词级起止) ===
    utterance_ranges_subword = [
        (utt_ranges[0][0], utt_ranges[-1][1])
        for utt_ranges in token_ranges_all
    ]
    # print(f"Dependency edges: {edge_type_sub.tolist().count(0)}")
    # print(f"Coreference edges: {edge_type_sub.tolist().count(1)}")
    # print(f"Emotion edges: {edge_type_sub.tolist().count(2)}")
    # print(f"SRL edges: {edge_type_sub.tolist().count(3)}")
    # print(f"Speaker edges: {edge_type_sub.tolist().count(4)}")
    # print(f"Turn edges: {edge_type_sub.tolist().count(5)}")

    return (
        torch.tensor(total_input_ids, dtype=torch.long),  # optional
        edge_index_sub,
        edge_type_sub,
        token_ranges_all,
        utterance_ranges_subword
    )
