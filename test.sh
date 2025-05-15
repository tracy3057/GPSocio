python test.py \
    --pretrain_ckpt pretrain_ckpt/seqrec_pretrain_ckpt.bin \
    --data_path  /data/liu323/gpsocio_framework/target_data\
    --num_train_epochs 20 \
    --batch_size 1 \
    --device 1 \
    --fp16 \
    --finetune_negative_sample_size -1