python -W ignore main.py \
    --root_path=/home/kimyh/python/project/bscr \
    --phase=phase_test \
    --dataset_name=dataset_test \
    --device_num=1 \
    --local_rank=0 \
    --purpose=generation \
    --epochs=100 \
    --batch_size=1 \
    --optimizer_name=Adam \
    --learning_rate=0.001 \
    --max_grad_norm=1 \
    --random_seed=42 \
    --num_workers=5 \
    --total_iter=100 \
    --warmup_iter=10 \
    --temporal_type=timeF \
    --embed_type=0 \
    --pred_len=12 \
    --factor=1 \
    --dropout_p=0.1 \
    --d_model=8 \
    --n_heads=4 \
    --d_ff=512 \
    --activation=gelu \
    --e_layer_num=4 \
    --d_layer_num=4 \
    --seq_len=600 \
    --moving_avg=25 \
    --pre_trained=None
    

    
    

#'ADC_BATT', 'BMSM1_TempInlet', 'BMSM2_TempInlet', 'BMSS1_TempInlet',
#'BMSS2_TempInlet', 'BMSM1_TempOutlet', 'BMSM2_TempOutlet',
#'BMSS1_TempOutlet', 'BMSS2_TempOutlet', 'BMSM1_SOC', 'BMSM2_SOC',
#'BMSS1_SOC', 'BMSS2_SOC', 
#'cal_ABS_DrivingAxlePressure_R', 'cal_ABS_DrivingAxlePressure_L
