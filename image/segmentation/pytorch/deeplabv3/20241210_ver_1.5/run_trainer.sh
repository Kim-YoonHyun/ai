python model_trainer.py \
    --root_path=/home/kimyh/python/project/2022/07_golf_cart \
    --root_dataset_path=/home/kimyh/python/project/2022/07_golf_cart \
    --phase=phase_test \
    --dataset_name=google_dataset_15 \
    --network=deeplabv3_resnet101 \
    --purpose=classification \
    --device_num=0 \
    --epochs=100 \
    --batch_size=8 \
    --max_grad_norm=1 \
    --loss_function_name=CrossEntropyLoss \
    --optimizer_name=AdamW \
    --scheduler_name=ExponentialLR \
    --gamma=0.98 \
    --learning_rate=2e-3 \
    --random_seed=42 \
    --shuffle= \
    --drop_last= \
    --num_workers=5 \
    --pin_memory=True \
    --pre_trained=None \
    --start_epoch=0
    

    