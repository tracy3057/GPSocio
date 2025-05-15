# Use distributed data parallel
CUDA_VISIBLE_DEVICES=1 python lightning_pretrain.py \
    --model_name_or_path allenai/longformer-base-4096 \
    --train_file /data/liu323/gpsocio_framework/pretrain_data/train.json\
    --dev_file /data/liu323/gpsocio_framework/pretrain_data/dev.json \
    --item_attr_file /data/liu323/gpsocio_framework/pretrain_data/meta_data.json \
    --output_dir /data/liu323/gpsocio_framework/output \
    --num_train_epochs 30 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8  \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --temp 0.05 \
    --device 1 \
    --fp16 \
    --fix_word_embedding