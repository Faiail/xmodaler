CUDA_VISIBLE_DEVICES=0 python3 train_net.py --config-file configs/video_caption/msvd/mplstm/mplstm.yaml --num-gpus 1 DATALOADER.MAX_FEAT_NUM 50 OUTPUT_DIR ./experiments/msvd-mplstm
