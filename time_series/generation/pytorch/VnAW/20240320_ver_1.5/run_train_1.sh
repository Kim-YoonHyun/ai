python -W ignore main.py \
    --root_path=/home/kimyh/python/project/bee \
    --phase=phase_test \
    --dataset_name=dataset_test_07/DPF/ECU_DOCGasTemperature_After \
    --network_name=res_lffnn \
    --purpose=generation \
    --device_num=1 \
    --epochs=50 \
    --batch_size=8 \
    --max_grad_norm=1 \
    --dropout_p=0.1 \
    --loss_function_name=MSE \
    --optimizer_name=Adam \
    --scheduler_name=CosineAnnealingLR \
    --learning_rate=0.001 \
    --random_seed=42 \
    --shuffle= \
    --drop_last= \
    --num_workers=5 \
    --pin_memory=True \
    --total_iter=100 \
    --warmup_iter=10 \
    --pre_trained=None \
    --d_model=512 \
    --d_ff=32 \
    --pred_len=1200 \
    --n_heads=1 \
    --embed_type=0 \
    --temporal_type=timeF \
    --enc_layer_num=1 \
    --enc_activation=relu \
    --dec_layer_num=1 \
    --dec_activation=gelu \

