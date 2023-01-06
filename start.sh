# pip install
pip3 install transformers==4.8.1
pip3 install timm
pip3 install ruamel.yaml
pip3 install opencv-python

echo "evaluate pretrain model first"
CUDA_VISIBLE_DEVICES="0" python3 Retrieval_MUGE.py \
--config ./configs/Retrieval_muge.yaml \
--output_dir /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEFResearch/muge_ngramwise_loss_ckpt09 \
--checkpoint /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEFResearch/ngramwise_loss_20220410/checkpoint_09.pth \
--text_encoder /apdcephfs/private_howardhwhou/video_search_product/pretrained_model/bert-base-chinese/ \
--evaluate

echo "finetuning pretrain model"
CUDA_VISIBLE_DEVICES="0" python3 Retrieval_MUGE.py \
--config ./configs/Retrieval_muge.yaml \
--output_dir /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEFResearch/muge_ngramwise_loss_ckpt09 \
--checkpoint /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEFResearch/ngramwise_loss_20220410/checkpoint_09.pth \
--text_encoder /apdcephfs/private_howardhwhou/video_search_product/pretrained_model/bert-base-chinese/ \
