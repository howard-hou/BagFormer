echo "Finetune tokenwise"
CUDA_VISIBLE_DEVICES="0" python3 train_muge.py \
--interaction tokenwise \
--output_dir output/retrieval_muge/tokenwise
