import argparse
import datetime
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from ruamel import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from models.loss import tokenwise_similarity_martix
from models.model_helper import EmbeddingBagHelperAutomaton
from models.model_retrieval import ALBEF
from models.model_retrieval_bagwise import BagFormer
from models.model_retrieval_tokenwise import ALBEF_tokenwise
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from MUGE_helper.dataset import create_dataloader
from MUGE_helper.evaluation import itm_eval, t2i_pred
from optim import create_optimizer
from scheduler import create_scheduler


def train(
    model,
    data_loader,
    optimizer,
    tokenizer,
    epoch,
    warmup_steps,
    device,
    scheduler,
    config,
    embedding_bag_helper,
):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "loss_itm", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_ita", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_twc", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, (image, text, idx) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(text, padding="max_length", max_length=25)
        embed_bag_offset, attn_mask = embedding_bag_helper.process(
            text_input, return_mask=True
        )
        embed_bag_offset = torch.LongTensor(embed_bag_offset).to(device)
        embed_bag_attn_mask = torch.LongTensor(attn_mask).to(device)
        embed_bag_input = dict(
            embed_bag_offset=embed_bag_offset, embed_bag_attn_mask=embed_bag_attn_mask
        )
        text_input = text_input.convert_to_tensors("pt").to(device)

        if epoch > 0 or not config["warm_up"]:
            alpha = config["alpha"]
        else:
            alpha = config["alpha"] * min(1, i / len(data_loader))

        if args.interaction == "bagwise":
            losses = model(
                image, text_input, embed_bag_input, alpha=alpha, idx=idx, config=config
            )
        else:
            losses = model(image, text_input, alpha=alpha, idx=idx)

        if len(losses) == 3:
            loss_ita, loss_itm, loss_twc = losses
            loss = loss_ita + loss_itm + loss_twc
        elif len(losses) == 2:
            loss_ita, loss_itm = losses
            loss = loss_ita + loss_itm
            loss_twc = loss
        else:
            raise "num of loss should be two or three"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_twc=loss_twc.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.6f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, embedding_bag_helper):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.texts
    num_img = len(data_loader.dataset.imgs)
    num_text = len(texts)
    if config["fusion_strategy"] == "late_fusion":
        text_atts, text_embeds, text_feats = encode_all_query(
            texts, model, tokenizer, device, config["batch_size_test"]
        )
    elif config["fusion_strategy"] == "early_fusion":
        text_atts, text_embeds, text_feats = encode_all_query_embedbag(
            texts,
            model,
            tokenizer,
            device,
            embedding_bag_helper,
            config["batch_size_test"],
        )
    else:
        raise f"fusion_strategy only support early fusion and late fusion"

    image_embeds, image_feats = encode_all_image(data_loader, model, device)

    # calc_sim_martix, recall
    if args.interaction == "cls_token":
        text_cls_embeds = text_embeds[:, 0, :]
        image_cls_embeds = image_embeds[:, 0, :]
        sim_i2t, sim_t2i = cls_token_similarity(image_cls_embeds, text_cls_embeds)
    elif args.interaction == "tokenwise":
        sim_i2t, sim_t2i = tokenwise_similarity_martix(text_embeds, image_embeds)
    elif args.interaction == "bagwise":
        text_embedbag_embeds = aggregate_embedbag(
            texts,
            text_feats,
            model,
            tokenizer,
            embedding_bag_helper,
            config["batch_size_test"],
        )
        sim_i2t, sim_t2i = tokenwise_similarity_martix(
            text_embedbag_embeds, image_embeds
        )
    else:
        raise f"model must be in [cls_token, tokenwise, bagwise]"

    rank_matrix_i2t = torch.full((num_img, num_text), -100.0).to(device)
    merge_matrix_i2t = torch.full((num_img, num_text), -100.0).to(device)

    start = 0
    end = sim_i2t.size(0)

    for i, sims in enumerate(metric_logger.log_every(sim_i2t[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)

        encoder_output = (
            image_feats[start + i].repeat(config["k_test"], 1, 1).float().to(device)
        )

        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            device
        )
        output = model.text_encoder(
            encoder_embeds=text_feats[topk_idx],
            attention_mask=text_atts[topk_idx],
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode="fusion",
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        rank_matrix_i2t[start + i, topk_idx] = score
        merge_matrix_i2t[start + i, topk_idx] = score + topk_sim

    rank_matrix_t2i = torch.full((num_text, num_img), -100.0).to(device)
    merge_matrix_t2i = torch.full((num_text, num_img), -100.0).to(device)

    start = 0
    end = sim_t2i.size(0)

    for i, sims in enumerate(metric_logger.log_every(sim_t2i[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        encoder_output = image_feats[topk_idx.cpu()].float().to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            device
        )
        output = model.text_encoder(
            encoder_embeds=text_feats[start + i].repeat(config["k_test"], 1, 1),
            attention_mask=text_atts[start + i].repeat(config["k_test"], 1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode="fusion",
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        rank_matrix_t2i[start + i, topk_idx] = score
        merge_matrix_t2i[start + i, topk_idx] = score + topk_sim

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return dict(
        recall_i2t=sim_i2t.cpu().numpy(),
        rank_i2t=rank_matrix_i2t.cpu().numpy(),
        merge_i2t=merge_matrix_i2t.cpu().numpy(),
        recall_t2i=sim_t2i.cpu().numpy(),
        rank_t2i=rank_matrix_t2i.cpu().numpy(),
        merge_t2i=merge_matrix_t2i.cpu().numpy(),
    )


def aggregate_embedbag(
    texts, text_feats, model, tokenizer, embedding_bag_helper, batch_size
):
    device = text_feats.device
    num_text = len(text_feats)
    embedbag_feats_all = []
    for i in range(0, num_text, batch_size):
        text = [text for text, _ in texts[i : min(num_text, i + batch_size)]]
        text_input = tokenizer(
            text, padding="max_length", truncation=True, max_length=25
        )
        embed_bag_offset, attn_mask = embedding_bag_helper.process(
            text_input, return_mask=True
        )
        embed_bag_offset = torch.LongTensor(embed_bag_offset).to(device)
        embed_bag_attn_mask = torch.LongTensor(attn_mask).to(device)
        batch_text_feats = text_feats[i : min(num_text, i + batch_size)]
        batch_size, seq_len, text_width = batch_text_feats.shape
        embedding_input = torch.arange(batch_size * seq_len, device=device)
        embedbag_feats = F.embedding_bag(
            embedding_input,
            batch_text_feats.view(-1, text_width),
            embed_bag_offset,
            mode="sum",
        ).view(batch_size, -1, text_width)
        embedbag_feats = F.normalize(model.text_proj(embedbag_feats), dim=-1)
        # mask cls and pad token
        embedbag_feats = embedbag_feats * embed_bag_attn_mask.unsqueeze(2)
        # pad to same length
        embedbag_seq_len = embedbag_feats.shape[1]
        embedbag_feats = F.pad(
            embedbag_feats,
            pad=(0, 0, 0, 25 - embedbag_seq_len, 0, 0),
            mode="constant",
            value=0,
        )
        embedbag_feats_all.append(embedbag_feats)
    embedbag_feats_all = torch.cat(embedbag_feats_all, dim=0)
    return embedbag_feats_all


def encode_all_image(data_loader, model, device):
    img_len = len(data_loader.dataset.imgs)
    image_feats = torch.zeros((img_len, 257, 768), dtype=torch.float16)
    image_embeds = []
    ptr = 0
    for image, img_id in data_loader:
        bs = len(image)
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        image_feat = image_feat.cpu().half()
        image_feats[ptr : ptr + bs] = image_feat
        image_embeds.append(image_embed.cpu())
        ptr += bs

    image_embeds = torch.cat(image_embeds, dim=0).to(device)
    return image_embeds, image_feats


def encode_all_query_embedbag(
    texts, model, tokenizer, device, embedding_bag_helper, text_bs=64
):
    text_feats = []
    text_embeds = []
    text_atts = []
    num_text = len(texts)
    text_len = 25
    for i in range(0, num_text, text_bs):
        text = [text for text, _ in texts[i : min(num_text, i + text_bs)]]
        text_input = tokenizer(
            text, padding="max_length", truncation=True, max_length=text_len
        )
        embed_bag_offset, attn_mask = embedding_bag_helper.process(
            text_input, return_mask=True
        )
        embed_bag_offset = torch.LongTensor(embed_bag_offset).to(device)
        embed_bag_attn_mask = torch.LongTensor(attn_mask).to(device)
        text_input = text_input.convert_to_tensors("pt").to(device)
        #
        text_embeddings = model.text_encoder.embeddings(input_ids=text_input.input_ids)
        batch_size, seq_len, text_width = text_embeddings.shape
        embedding_input = torch.arange(batch_size * seq_len, device=device)
        text_embed_bags = F.embedding_bag(
            embedding_input,
            text_embeddings.view(-1, text_width),
            embed_bag_offset,
            mode="sum",
        ).view(batch_size, -1, text_width)

        text_output = model.text_encoder(
            inputs_embeds=text_embed_bags,
            attention_mask=embed_bag_attn_mask,
            mode="text",
        )
        text_feat = text_output.last_hidden_state
        text_feat = F.pad(
            text_feat,
            pad=(0, 0, 0, text_len - text_feat.shape[1], 0, 0),
            mode="constant",
            value=0,
        )
        text_embed = F.normalize(model.text_proj(text_feat), dim=-1)

        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        embed_bag_attn_mask = F.pad(
            embed_bag_attn_mask,
            pad=(0, text_len - embed_bag_attn_mask.shape[1]),
            mode="constant",
            value=0,
        )
        text_atts.append(embed_bag_attn_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    return text_atts, text_embeds, text_feats


def encode_all_query(texts, model, tokenizer, device, text_bs=64):
    text_feats = []
    text_embeds = []
    text_atts = []
    num_text = len(texts)
    for i in range(0, num_text, text_bs):
        text = [text for text, _ in texts[i : min(num_text, i + text_bs)]]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=25,
            return_tensors="pt",
        ).to(device)
        text_output = model.text_encoder(
            text_input.input_ids, attention_mask=text_input.attention_mask, mode="text"
        )
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    return text_atts, text_embeds, text_feats


def cls_token_similarity(image_embeds, text_embeds):
    sim_i2t = image_embeds @ text_embeds.t()
    sim_t2i = text_embeds @ image_embeds.t()
    return sim_i2t, sim_t2i


def output_prediction(test_eval_dict, test_loader, key):
    pred_list = t2i_pred(
        test_eval_dict[key],
        test_loader.dataset.texts,
        test_loader.dataset.img_idx2img_id,
    )
    test_outpath = os.path.join(args.output_dir, f"{key}.jsonl")
    with open(test_outpath, "w", encoding="utf-8") as w:
        for p in pred_list:
            p = json.dumps(p, ensure_ascii=False)
            w.write(f"{p}\n")


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating muge retrieval dataset")

    train_loader, val_loader, test_loader = create_dataloader(
        "multimodal_retrieval", config
    )

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    embedding_bag_helper = EmbeddingBagHelperAutomaton(
        tokenizer, config["entity_dict_path"], masked_token=["[CLS]", "[PAD]"]
    )

    #### Model ####
    print("Creating model")
    if args.interaction == "cls_token":
        model = ALBEF(
            config=config, text_encoder=args.text_encoder, tokenizer=tokenizer
        )
        print("create ALBEF cls token model")
    elif args.interaction == "tokenwise":
        model = ALBEF_tokenwise(
            config=config, text_encoder=args.text_encoder, tokenizer=tokenizer
        )
        print("create ALBEF_tokenwise model")
    elif args.interaction == "bagwise":
        model = BagFormer(
            config=config, text_encoder=args.text_encoder, tokenizer=tokenizer
        )
        print("create BagFormer model")
    else:
        raise f"similarity_metric must be in [cls_token, tokenwise, bagwise]"

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], model.visual_encoder
        )
        state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
        )
        state_dict["visual_encoder_m.pos_embed"] = m_pos_embed_reshaped

        for key in list(state_dict.keys()):
            if "bert" in key:
                encoder_key = key.replace("bert.", "")
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        del state_dict["image_queue"]
        del state_dict["text_queue"]
        msg = model.load_state_dict(state_dict, strict=False)

        print("load checkpoint from %s" % args.checkpoint)
        print(msg)

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    arg_opt = utils.AttrDict(config["optimizer"])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config["schedular"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(
                model,
                train_loader,
                optimizer,
                tokenizer,
                epoch,
                warmup_steps,
                device,
                lr_scheduler,
                config,
                embedding_bag_helper,
            )

        eval_dict = evaluation(
            model_without_ddp,
            val_loader,
            tokenizer,
            device,
            config,
            embedding_bag_helper,
        )

        if utils.is_main_process():

            recall_val_result = itm_eval(
                eval_dict["recall_i2t"],
                eval_dict["recall_t2i"],
                val_loader.dataset.txt2img,
                val_loader.dataset.img2txt,
            )
            
            rank_val_result = itm_eval(
                eval_dict["rank_i2t"],
                eval_dict["rank_t2i"],
                val_loader.dataset.txt2img,
                val_loader.dataset.img2txt,
            )
            
            merge_val_result = itm_eval(
                eval_dict["merge_i2t"],
                eval_dict["merge_t2i"],
                val_loader.dataset.txt2img,
                val_loader.dataset.img2txt,
            )
            

            if args.evaluate:
                log_stats = {
                    **{
                        f"recall_val_{k}": round(v, 3)
                        for k, v in recall_val_result.items()
                    },
                    **{
                        f"rank_val_{k}": round(v, 3) for k, v in rank_val_result.items()
                    },
                    **{
                        f"merge_val_{k}": round(v, 3)
                        for k, v in merge_val_result.items()
                    },
                    "epoch": epoch,
                }
                with open(
                    os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8"
                ) as f:
                    f.write(f"evaluate start\n")
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{
                        f"recall_val_{k}": round(v, 3)
                        for k, v in recall_val_result.items()
                    },
                    **{
                        f"rank_val_{k}": round(v, 3) for k, v in rank_val_result.items()
                    },
                    **{
                        f"merge_val_{k}": round(v, 3)
                        for k, v in merge_val_result.items()
                    },
                    "epoch": epoch,
                }
                with open(
                    os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

                if merge_val_result["img_r_mean"] > best:
                    save_obj = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "config": config,
                        "epoch": epoch,
                    }
                    best_model_path = os.path.join(
                        args.output_dir, "checkpoint_best.pth"
                    )
                    torch.save(save_obj, best_model_path)
                    best = merge_val_result["img_r_mean"]
                    best_epoch = epoch

        if args.evaluate:
            break

        lr_scheduler.step(epoch + warmup_steps + 1)
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
            f.write("best epoch: %d" % best_epoch + "\n")

    print(f"{datetime.datetime.now()}, Start testing")
    if not args.evaluate:
        ckpt = torch.load(best_model_path)
        model_without_ddp.load_state_dict(ckpt["model"])
    test_eval_dict = evaluation(
        model_without_ddp, test_loader, tokenizer, device, config, embedding_bag_helper
    )
    output_prediction(test_eval_dict, test_loader, "merge_t2i")
    output_prediction(test_eval_dict, test_loader, "recall_t2i")
    output_prediction(test_eval_dict, test_loader, "rank_t2i")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_muge.yaml")
    parser.add_argument("--output_dir", default="output/retrieval_muge")
    parser.add_argument(
        "--checkpoint", default="pretrained_model/pretrained_checkpoint.pth"
    )
    parser.add_argument("--text_encoder", default="bert-base-chinese")
    parser.add_argument("--interaction", default="bagwise")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    main(args, config)
