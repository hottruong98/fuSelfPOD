# NOTE: 
# mAP: 0.3074
# mATE: 0.8311
# mASE: 0.7138
# mAOE: 1.5040
# mAVE: 1.3074
# mAAE: 0.4258
# NDS: 0.2566
GPUS=8
PORT=1998
CONFIG=projects/configs/exp/uvtr_convnext_s_vs0.1_c128_finetune.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT ./extra_tools/train.py $CONFIG  --launcher pytorch