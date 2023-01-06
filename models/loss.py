import torch
import torch.nn.functional as F


def sparse_cls_token_similarity(image_embeds, text_embeds, topk=256):
    sim_i2t_kv = []
    text_embeds_t = text_embeds.t()
    for image_embed in image_embeds:
        topk_sim, topk_idx = (image_embed @ text_embeds_t).topk(k=topk, dim=0)
        ind2sim = {k.item(): v.item() for k, v in zip(topk_idx, topk_sim)}
        sim_i2t_kv.append(ind2sim)

    sim_t2i_kv = []
    image_embeds_t = image_embeds.t()
    for text_embed in text_embeds:
        topk_sim, topk_idx = (text_embed @ image_embeds_t).topk(k=topk, dim=0)
        ind2sim = {k.item(): v.item() for k, v in zip(topk_idx, topk_sim)}
        sim_t2i_kv.append(ind2sim)
    return sim_i2t_kv, sim_t2i_kv


def sparse_image2text_sim_martix(text_feats, image_feats, topk=256):
    num_image = image_feats.shape[0]
    sim_i2t = []
    for i in range(num_image):
        row_sim = tokenwise_similarity(image_feats[i], text_feats)
        topk_sim, topk_idx = row_sim.topk(k=topk, dim=0)
        ind2sim = {k.item(): v.item() for k, v in zip(topk_idx, topk_sim)}
        sim_i2t.append(ind2sim)
    return sim_i2t


def sparse_text2image_sim_martix(text_feats, image_feats, topk=256):
    num_text = text_feats.shape[0]
    sim_t2i = []
    for i in range(num_text):
        row_sim = tokenwise_similarity(text_feats[i], image_feats)
        topk_sim, topk_idx = row_sim.topk(k=topk, dim=0)
        ind2sim = {k.item(): v.item() for k, v in zip(topk_idx, topk_sim)}
        sim_t2i.append(ind2sim)
    return sim_t2i


def sparse_tokenwise_similarity_martix(text_feats, image_feats, topk=256):
    sim_i2t = sparse_image2text_sim_martix(text_feats, image_feats, topk=topk)
    sim_t2i = sparse_text2image_sim_martix(text_feats, image_feats, topk=topk)
    return sim_i2t, sim_t2i


def image2text_sim_martix(text_feats, image_feats):
    num_text, num_image = text_feats.shape[0], image_feats.shape[0]
    sim_i2t = torch.zeros((num_image, num_text)).to(text_feats.device)
    for i in range(num_image):
        row_sim = tokenwise_similarity(image_feats[i], text_feats)
        sim_i2t[i] = row_sim
    return sim_i2t


def text2image_sim_martix(text_feats, image_feats):
    num_text, num_image = text_feats.shape[0], image_feats.shape[0]
    sim_t2i = torch.zeros((num_text, num_image)).to(text_feats.device)
    for i in range(num_text):
        row_sim = tokenwise_similarity(text_feats[i], image_feats)
        sim_t2i[i] = row_sim
    return sim_t2i


def tokenwise_similarity_martix(text_feats, image_feats):
    sim_i2t = image2text_sim_martix(text_feats, image_feats)
    sim_t2i = text2image_sim_martix(text_feats, image_feats)
    return sim_i2t, sim_t2i


def tokenwise_similarity_loss(text_feats, image_feats):
    sim_i2t, sim_t2i = tokenwise_similarity_martix(text_feats, image_feats)

    i2t_targets = torch.zeros_like(sim_i2t)
    i2t_targets.fill_diagonal_(1)

    t2i_targets = torch.zeros_like(sim_t2i)
    t2i_targets.fill_diagonal_(1)

    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * i2t_targets, dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * t2i_targets, dim=1).mean()
    return (loss_i2t + loss_t2i) / 2


def tokenwise_similarity(Q, D, similarity_metric='cosine'):
    if similarity_metric == 'cosine':
        return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

    assert similarity_metric == 'l2'
    return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)


def ngramwise_similarity_loss(text_feats, image_feats):
    '''consider text with ngram feats, image with token feats'''
    batch_size, num_ngram = text_feats.shape[:2]
    sim_i2t = torch.zeros((batch_size, batch_size)).to(text_feats.device)
    for i in range(batch_size):
        sim_i2t[i] = ngramwise_similarity(image_feats[i], text_feats)

    sim_t2i = torch.zeros((batch_size, batch_size)).to(text_feats.device)
    for i in range(batch_size):
        ngram_list = []
        for j in range(num_ngram):
            ngram = (text_feats[i][j] @ image_feats.permute(0, 2, 1))
            ngram_list.append(ngram.unsqueeze(-1))
        ngram_sim_matrix = torch.cat(ngram_list, dim=-1).permute(0, 3, 1, 2)  # (N, ngram, seq, seq)
        sim_t2i[i] = ngram_sim_matrix.max(1).values.max(1).values.sum(1)

    sim_targets = torch.zeros_like(sim_i2t)
    sim_targets.fill_diagonal_(1)

    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
    return (loss_i2t + loss_t2i) / 2


def ngramwise_similarity(Q, D):
    '''Q: (seq_len, dim), D: (N, ngram, seq_len, dim)'''
    return (Q @ D.permute(0, 1, 3, 2)).max(1).values.max(1).values.sum(1)
