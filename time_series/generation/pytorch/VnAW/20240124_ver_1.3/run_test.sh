python -W ignore test.py \
    --root_path=/home/kimyh/python/project/VnAW \
    --trained_model_path1=/home/kimyh/python/project/VnAW/trained_model/phase_test1/dataset_03/0002 \
    --trained_model_path2=A/ECU_BatteryVolt \
    --trained_model_name=epoch0002 \
    --test_data_path=/home/kimyh/python/project/VnAW/datasets/dataset_03/A/ECU_BatteryVolt/train \
    --shuffle= \
    --drop_last= \
    --num_workers=5 \
    --pin_memory=True
    

    # --trained_model_path1=/home/kimyh/python/project/transformer/trained_model/phase_test1/dataset_03/0000 \
    # --trained_model_path2=A/ECU_InjectionGasTemp \
    # --trained_model_name=epoch0100 \
    