python main.py \
    --root_path=/home/kimyh/python/project/kca/KoBERT-NER \
    --phase=phase_01 \
    --dataset_name=dataset_14_aug \
    --tokenizer=None \
    --device_num=2 \
    --epochs=30 \
    --batch_size=64 \
    --max_seq_len=512 \
    --random_seed=42 \
    --learning_rate=5e-5 \
    --weight_decay=0.0 \
    --gradient_accumulation_steps=1 \
    --adam_epsilon=1e-8 \
    --max_grad_norm=1.0 \
    --max_steps=-1 \
    --warmup_steps=0