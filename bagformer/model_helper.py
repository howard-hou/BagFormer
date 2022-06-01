import torch
import torch.nn as nn


def tokenwise_similarity_martix(text_feats, image_feats):
    num_text, num_image = text_feats.shape[0], image_feats.shape[0]
    sim_i2t = torch.zeros((num_image, num_text)).to(text_feats.device)
    for i in range(num_image):
        row_sim = tokenwise_similarity(image_feats[i], text_feats)
        sim_i2t[i] = row_sim

    sim_t2i = torch.zeros((num_text, num_image)).to(text_feats.device)
    for i in range(num_text):
        row_sim = tokenwise_similarity(text_feats[i], image_feats)
        sim_t2i[i] = row_sim
    return sim_i2t, sim_t2i

def tokenwise_similarity(Q, D, similarity_metric='cosine'):
    if similarity_metric == 'cosine':
        return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

    assert similarity_metric == 'l2'
    return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)


class EmbeddingBagHelper:
    def __init__(self, tokenizer, dict_path):
        self.tokenizer = tokenizer
        self.entity_ids_dict = self.load_dict(dict_path)

    def load_dict(self, dict_path):
        entity_ids_dict = {}
        for line in open(dict_path, encoding="utf8"):
            entity_ids = self.tokenizer.encode(line.strip(), add_special_tokens=False)
            entity_ids_dict[tuple(entity_ids)] = line.strip()
        return entity_ids_dict

    def backward_maximum_matching(self, input_ids):
        idx, idy = 0, len(input_ids)
        offsets = []
        tmp_offset_len = 0
        while idy > 0:
            while idx < idy:
                if tuple(input_ids[idx:idy]) in self.entity_ids_dict:
                    offsets.append((idx, idy))
                    break
                idx = idx + 1
            if len(offsets) != tmp_offset_len:
                idy = idx
                idx = 0
                tmp_offset_len = len(offsets)
            else:
                idx = 0
                idy = idy - 1
        return offsets

    def pad_offsets(self, offsets_sort, input_len):
        # step 0: if offset list is empty, return range
        if not offsets_sort:
            return list(range(input_len))
        # step 1: pad index before first offset
        first_offset = offsets_sort[0]
        offsets_pad = [i for i in range(first_offset[0])]
        # step 2: pad index between
        group_size = len(offsets_sort)
        for i in range(len(offsets_sort)):
            offsets_pad.append(offsets_sort[i][0])
            offsets_pad.append(offsets_sort[i][1])
            if i + 1 < group_size:
                first_end = offsets_sort[i][1]
                second_start = offsets_sort[i+1][0]
                for j in range(first_end+1, second_start):
                    offsets_pad.append(j)
        # step 3: pad index last
        last_offset = offsets_sort[-1]
        for k in range(last_offset[1]+1, input_len):
            offsets_pad.append(k)
        return offsets_pad

    def get_offsets(self, input_ids):
        offsets = self.backward_maximum_matching(input_ids)
        offsets_sort = sorted(offsets, key=lambda x: x[0])
        return self.pad_offsets(offsets_sort, len(input_ids))

    def process(self, text_input, return_mask=False):
        input_offsets = [self.get_offsets(i) for i in text_input.input_ids]
        min_len = min([len(offset) for offset in input_offsets])
        # truncation
        input_offsets_truncated = [offset[:min_len] for offset in input_offsets]
        # flatten to 1d
        input_offsets_flatten = []
        text_len = len(text_input.input_ids[0])
        for i, offset in enumerate(input_offsets_truncated):
            input_offsets_flatten.extend([o + i*text_len for o in offset])
        # return
        if return_mask:
            attn_mask_truncated = [mask[:min_len] for mask in text_input.attention_mask]
            return input_offsets_flatten, attn_mask_truncated
        return input_offsets_flatten
