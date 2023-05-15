python -W ignore main.py \
    --root_path=/home/kimyh/python/ai/time_series/generation/pytorch/transformer \
    --group=C \
    --dataset_name=dataset01 \
    --column_type=S \
    --train_p=0.9 \
    --local_rank=0 \
    --ddp=no \
    --purpose=generation \
    --epochs=30 \
    --batch_size=32 \
    --device=2 \
    --optimizer_name=Adam \
    --learning_rate=0.001 \
    --max_grad_norm=1 \
    --num_workers=5 \
    --total_iter=100 \
    --warmup_iter=10 \
    --retrain= \
    --embed=timeF \
    --embed_type=0 \
    --pred_len=600 \
    --factor=1 \
    --dropout=0.1 \
    --d_model=256 \
    --n_heads=8 \
    --d_ff=512 \
    --activation=gelu \
    --e_layers=1 \
    --d_layers=1 \
    --seq_len=600 \
    --moving_avg=25 \

    
    

#'ADC_BATT', 'BMSM1_TempInlet', 'BMSM2_TempInlet', 'BMSS1_TempInlet',
#'BMSS2_TempInlet', 'BMSM1_TempOutlet', 'BMSM2_TempOutlet',
#'BMSS1_TempOutlet', 'BMSS2_TempOutlet', 'BMSM1_SOC', 'BMSM2_SOC',
#'BMSS1_SOC', 'BMSS2_SOC', 
#'cal_ABS_DrivingAxlePressure_R', 'cal_ABS_DrivingAxlePressure_L
