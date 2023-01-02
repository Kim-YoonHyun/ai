python main.py \
    /home/kimyh/python/ai/image/classification/project/smartfarm \
    phase_test \
    disease_tomato \
    --network=efficientnet_b4 \
    --batch_size=16 \
    --num_workers=1 \
    --device_num=0 \
    --epochs=5 \
    --optimizer_name=AdamW \
    --loss_name=CrossEntropyLoss \
    --scheduler_name=ExponentialLR \
    --learning_rate=2e-3 \
    --gamma=0.98 \
    --warmup_ratio=0.1 \
    --max_grad_norm=1 \
    --trained_weight=1st_phase/beef_hsv_4part_model/effnet/batch16_epoch0191/weight.pt \
    --start_epoch=192 \
    --random_seed=42
