# install L5
CUR_DIR=$(cd $(dirname $0); pwd)
INSTALL_DIR=/usr/local/lib    
cp /apdcephfs/private_howardhwhou/l5agent-bit64-4.3.0.tar ${INSTALL_DIR}
cd ${INSTALL_DIR} &&\
    tar -xvf l5agent-bit64-4.3.0.tar &&\
    rm l5agent-bit64-4.3.0.tar

cd l5agent-bit64-4.3.0 && ./install.sh && cd ${CUR_DIR}

echo "first test L5agent"
/usr/local/lib/l5agent-bit64-4.3.0/bin/L5GetRoute1 2055361 65536 1
sleep 5s
echo "second test L5agent"
/usr/local/lib/l5agent-bit64-4.3.0/bin/L5GetRoute1 2055361 65536 1

# pip install
pip3 install transformers==4.8.1
pip3 install timm
pip3 install ruamel.yaml
pip3 install opencv-python
pip3 install -U cos-python-sdk-v5

# finetune
CUDA_VISIBLE_DEVICES="0" python3 -m torch.distributed.launch --nproc_per_node=1 --use_env RP2K.py \
--config ./configs/RP2K.yaml \
--output_dir /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEF/100buckets_RP2K/finetune \
--text_encoder /apdcephfs/private_howardhwhou/video_search_product/pretrained_model/bert-base-chinese/ \
--checkpoint /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEF/Pretrain_commodity_100buckets/checkpoint_13.pth
# freeze params
CUDA_VISIBLE_DEVICES="0" python3 -m torch.distributed.launch --nproc_per_node=1 --use_env RP2K.py \
--config ./configs/RP2K.yaml \
--output_dir /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEF/100buckets_RP2K/freeze \
--text_encoder /apdcephfs/private_howardhwhou/video_search_product/pretrained_model/bert-base-chinese/ \
--checkpoint /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEF/Pretrain_commodity_100buckets/checkpoint_13.pth \
--freeze
# train from scratch
CUDA_VISIBLE_DEVICES="0" python3 -m torch.distributed.launch --nproc_per_node=1 --use_env RP2K.py \
--config ./configs/RP2K.yaml \
--output_dir /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEF/100buckets_RP2K/from_scratch \
--text_encoder /apdcephfs/private_howardhwhou/video_search_product/pretrained_model/bert-base-chinese/ \
--from_scratch
