python main.py \
    /home/kimyh/python/ai/text/classification/project/complaints \
    phase_1st \
    week1_7_without_unk_drop500_limit1000_integrated.csv \
    교통안전,수질,식품의약품,위생,의료,자연재해,화재 \
    --network=kobert \
    --batch_size=32 \
    --max_len=512 \
    --epochs=100 \
    --device_num=0 \
    --pre_trained=skt/kobert-base-v1 \
    --optimizer_name=AdamW \
    --loss_name=CrossEntropyLoss \
    --scheduler_name=cosine_warmup \
    --learning_rate=2e-3 \
    --trained_weight=weight_path/weight.pt \
    --start_epoch=1 \