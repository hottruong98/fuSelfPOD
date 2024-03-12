GPUS=8
PORT=1998
CONFIG=projects/configs/exp/uvtr_convnext_s_vs0.1_c128_pretrain_full.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT  ./extra_tools/train.py $CONFIG  --launcher pytorch --no-validate