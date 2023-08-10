python main.py \
    --root_path=/home/kimyh/python/project/datom/preprocessor \
    --phase=phase_test \
    --dataset_name=dataset_24 \
    --purpose=classification \
    --device_num=0 \
    --epochs=15 \
    --batch_size=16 \
    --max_grad_norm=1 \
    --dropout_p=0.5 \
    --loss_function_name=CrossEntropyLoss \
    --learning_rate=5e-5 \
    --gamma=0.98 \
    --shuffle=\
    --drop_last=\
    --num_workers=1 \
    --pin_memory=\
    --max_len=512 \
    --warmup_ratio=0.1 \
    --pad=True \
    --pair=\

    # --network=kobert \
    
    # --pre_trained=skt/kobert-base-v1 \
    
    # --amp=\
    # --retrain=\
    # --trained_weight=weight_path/weight.pt \
    # --start_epoch=1 \
    # --random_seed=42