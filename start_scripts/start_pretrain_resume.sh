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


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node=8 --use_env Pretrain_commodity.py \
--config ./configs/Pretrain_commodity.yaml \
--output_dir /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEFResearch/tokenwise_loss_20220329 \
--checkpoint /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEFResearch/tokenwise_loss_20220329/checkpoint_09.pth \
--text_encoder /apdcephfs/private_howardhwhou/video_search_product/pretrained_model/bert-base-chinese/ \
--resume

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node=8 --use_env Pretrain_commodity.py \
--config ./configs/Pretrain_commodity.yaml \
--output_dir /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEFResearch/tokenwise_loss_20220329 \
--checkpoint /apdcephfs/private_howardhwhou/video_search_product/saved_model/ALBEFResearch/tokenwise_loss_20220329/checkpoint_10.pth \
--text_encoder /apdcephfs/private_howardhwhou/video_search_product/pretrained_model/bert-base-chinese/ \
--resume

