import torch
import torch.nn as nn


def nan_to_num(x, num=0):
    x = torch.where(torch.isnan(x), torch.full_like(x, num), x)
    x = torch.where(torch.isinf(x), torch.full_like(x, num), x)
    return x


def cross_entropy_mp(input, target, eps=1e-7):
    input = torch.softmax(input.float(), dim=1) + eps
    input = torch.log(input)
    loss = nn.NLLLoss()(input, target)
    return loss


class EmbeddingBagHelper:
    def __init__(self, tokenizer, dict_path, masked_token=None):
        if masked_token is None:
            masked_token = ["[PAD]"]
        self.tokenizer = tokenizer
        self.entity_ids_dict = self.load_dict(dict_path)
        self.masked_ids = [[i] for i in tokenizer.convert_tokens_to_ids(masked_token)]

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
                second_start = offsets_sort[i + 1][0]
                for j in range(first_end + 1, second_start):
                    offsets_pad.append(j)
        # step 3: pad index last
        last_offset = offsets_sort[-1]
        for k in range(last_offset[1] + 1, input_len):
            offsets_pad.append(k)
        return offsets_pad

    def get_offsets(self, input_ids):
        offsets = self.backward_maximum_matching(input_ids)
        offsets_sort = sorted(offsets, key=lambda x: x[0])
        return self.pad_offsets(offsets_sort, len(input_ids))

    def get_embed_bag_input_ids(self, input_ids, embed_bag_offset):
        embed_bag_input_ids = []
        for i, start in enumerate(embed_bag_offset):
            if i + 1 < len(embed_bag_offset):
                end = embed_bag_offset[i + 1]
                embed_bag_input_ids.append(input_ids[start:end])
            else:
                embed_bag_input_ids.append(input_ids[start:])
        return embed_bag_input_ids

    def get_embed_bag_attn_mask(self, one_input_ids):
        one_attn_mask = []
        for input_id in one_input_ids:
            if input_id not in self.masked_ids:
                one_attn_mask.append(1)
            else:
                one_attn_mask.append(0)
        return one_attn_mask

    def process(self, text_input, return_mask=False):
        input_offsets = [self.get_offsets(i) for i in text_input.input_ids]
        min_len = min([len(offset) for offset in input_offsets])
        # truncation
        input_offsets_truncated = [offset[:min_len] for offset in input_offsets]
        # flatten to 1d
        input_offsets_flatten = []
        text_len = len(text_input.input_ids[0])
        for i, offset in enumerate(input_offsets_truncated):
            input_offsets_flatten.extend([o + i * text_len for o in offset])
        # return
        if return_mask:
            embed_bag_input_ids = [self.get_embed_bag_input_ids(i, o) for i, o in zip(
                text_input.input_ids, input_offsets)]
            attn_mask_list = [self.get_embed_bag_attn_mask(i) for i in embed_bag_input_ids]
            attn_mask_truncated = [mask[:min_len] for mask in attn_mask_list]
            return input_offsets_flatten, attn_mask_truncated
        return input_offsets_flatten
