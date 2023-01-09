echo "Evaluate BagFormer"
CUDA_VISIBLE_DEVICES="0" python3 train_muge.py \
--checkpoint output/retrieval_muge/bagformer/checkpoint_best.pth
--interaction bagwise \
--output_dir output/retrieval_muge/bagformer \
--evaluate
