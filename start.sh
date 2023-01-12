echo "Finetune cls_token"
CUDA_VISIBLE_DEVICES="0" python3 train_muge.py \
--interaction cls_token \
--output_dir output/retrieval_muge/cls_token
