GPUS=4
PORT=1998
CONFIG=projects/configs/unipad_abl/finetune_cfg_full.py
CHECKPOINT=work_dirs/finetune_cfg_full/epoch_12.pth
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT ./extra_tools/test.py $CONFIG $CHECKPOINT --eval mAP  