GPUS=4
PORT=1998
CONFIG=projects/configs/exp/uvtr_convnext_s_vs0.1_c128_finetune.py
CHECKPOINT=work_dirs/uvtr_convnext_s_vs0.1_c128_finetune/epoch_12.pth
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT ./extra_tools/test.py $CONFIG $CHECKPOINT --eval mAP