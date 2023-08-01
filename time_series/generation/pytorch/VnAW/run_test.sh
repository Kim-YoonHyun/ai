python -W ignore test.py \
    --root_path=/home/kimyh/python/project/transformer \
    --trained_model_path=/home/kimyh/python/project/transformer/trained_model/ttt/0001 \
    --trained_model_name=epoch1000 \
    --test_data_name=ttt \
    --device_num=0 \
    --shuffle= \
    --drop_last= \
    --num_workers=5 \
    --pin_memory=True
    