python model_test.py \
    --root_path=/home/kimyh/python/project/kca/KoBERT-NER \
    --trained_model_path=/home/kimyh/python/project/kca/KoBERT-NER/trained_model/phase_01/dataset_17_custom_no_continue_aug/0000 \
    --trained_model=epoch0018 \
    --device_num=2 \
    --stan_num=200 \
    --dummy_label=True \
    --test_data=20231211_customleft.jsonl \
    --result_save_path=/home/kimyh/python/project/kca/KoBERT-NER/result
