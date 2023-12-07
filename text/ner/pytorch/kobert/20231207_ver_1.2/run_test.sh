python model_test.py \
    --root_path=/home/kimyh/python/project/kca/KoBERT-NER \
    --trained_model_path=/home/kimyh/python/project/kca/KoBERT-NER/trained_model/phase_01/dataset_13/0000 \
    --trained_model=epoch0006 \
    --stan_num=200 \
    --dummy_label=True \
    --test_data=token_prodigy_out.jsonl
