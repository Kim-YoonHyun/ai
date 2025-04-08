'''
https://llama.meta.com/llama-downloads
-> 위 사이트로 이동해서 진행

pip install llama-stack
llama model download --source meta --model-id Llama3.1-8B
https://llama3-1.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZWhtYTU2Nm9tOTZlazVxY3Q0ODcwNHB2IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTEubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczOTU5MzUzN319fV19&Signature=AExKRbKCQ56Y4GxVepbwl74jqKrhkhbIhZXmvtSrwA2aHJVYjf05N%7EDiCmMYxFK4cdnJHqMuAbzMuOIWW8Xr2GsUi4A1WvCbY2xnNm22wvYvpkldhXmGuD-d3DZCDu2hEsz2YQt9d9xi55C9THAyPkqZLhCQWplOn3HagsQ5XZKLYyQHsNSR2De%7EU2h-d0JgYK69DM54UuGLYoIwrVhHzUiVMj8QQwxRbCxBHE1g3bY8j6fghkNRVWzNF5vjYS30EzElUCfWhhgUA4SbsX1G7C3%7EoKKtw9KIySxAybVDUqSTVQ4eRQsyNNnkG2h6MemS7O%7EeOz2CQb9KsOlBC1GtHQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1903401000192281
'''

'''
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
pip3 install torch torchvision torchaudio
pip install transformers
pip install accelerate>=0.26.0
pip install datasets
pip install peft
'''
import torch
import transformers 

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

import warnings
warnings.filterwarnings('ignore')


model_name = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=512)
    
    inputs["labels"] = outputs["input_ids"]  # labels 추가
    return inputs

# def tokenize_function(examples):
#     return tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)


def main():
    # 모델
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",  # GPU 사용
        torch_dtype="auto"   # 적절한 데이터 타입 설정 (FP16, BF16 등)
    )
    print(model)
    
    # 데이터셋
    dataset = load_dataset("json", data_files="train_data.json")
    
    # 토큰화
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # fine-tuning 설정
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=32,  
        target_modules=["q_proj", "v_proj"],  
        lora_dropout=0.1,  
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 학습 설정    
    training_args = TrainingArguments(
        output_dir="./llama3_finetuned",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        fp16=True
    )
    
    # 학습 진행    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"] if "test" in tokenized_datasets else None
    )
    trainer.train()
    
    # 모델 저장 및 사용
    model.save_pretrained("./llama3_finetuned")
    tokenizer.save_pretrained("./llama3_finetuned")

    # 모델 로드 후 사용
    finetuned_model = AutoModelForCausalLM.from_pretrained("./llama3_finetuned")
    pipe = pipeline("text-generation", model=finetuned_model, tokenizer=tokenizer)

    result = pipe("질문을 입력하세요", max_length=200)
    print(result)



# def main2():
#     pipeline = transformers.pipeline(
#         "text-generation",
#         model=model_id,
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         device_map="auto",
#         token = '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'
#     )

#     messages = [
#         {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#         {"role": "user", "content": "Who are you?"},
#     ]

#     terminators = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = pipeline(
#         messages,
#         max_new_tokens=256,
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#     )
#     print(outputs[0]["generated_text"][-1])


if __name__ == '__main__':
    main()