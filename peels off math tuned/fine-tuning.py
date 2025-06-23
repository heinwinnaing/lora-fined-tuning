import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

df = pd.read_csv('data/train_data.csv', index_col=0)

train_size = int(len(df) * 0.8)
valid_size = int(len(df) * 0.2)

dataset_dic = DatasetDict({
    'train': df[:train_size],
    'eval': df[train_size:]
})

model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


def tokenize(batch):
    texts = [
        f'### Instruction:\n{question}\n### Response:\n{answer}'
        for question, answer in zip(batch['question'], batch['answer'])
    ]
    tokens = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=256
    )
    tokens['labels'] = tokens['input_ids']
    return tokens


train_dataset = Dataset.from_dict(dataset_dic['train']).map(
    tokenize,
    batched=True,
    remove_columns=['question', 'answer']
)

eval_dataset = Dataset.from_dict(dataset_dic['eval']).map(
    tokenize,
    batched=True,
)

training_args = TrainingArguments(
    output_dir='./results/tinyllama-lora-tuned',
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    num_train_epochs=4,
    learning_rate=2e-5,
    save_strategy='epoch',
    logging_steps=4,
    report_to = 'none',
    use_cpu=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer
)

print('Training started...')
trainer.train()
print('Training completed')
