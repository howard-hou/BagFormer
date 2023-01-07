echo "Finetune BagFormer"
CUDA_VISIBLE_DEVICES="0" python3 train_muge.py \
--interaction bagwise \
--output_dir output/retrieval_muge/bagformer

