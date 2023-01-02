python main.py \
    /home/kimyh/python/project/golf_cart \
    phase_2nd \
    google_dataset_04 \
    --network=deeplabv3_resnet101 \
    --batch_size=8 \
    --shuffle=\
    --num_workers=5 \
    --pin_memory=\
    --drop_last=\
    --device_num=0 \
    --epochs=100 \
    --optimizer_name=AdamW \
    --loss_name=CrossEntropyLoss \
    --scheduler_name=ExponentialLR \
    --learning_rate=2e-3 \
    --gamma=0.98 \
    --warmup_ratio=0.1 \
    --amp=\
    --max_grad_norm=1 \
    --retrain=True \
    --trained_weight=phase_1st/vworld_07_reset_001_model/0000/epoch0085/weight.pt \
    --start_epoch=0 \
    --random_seed=42   