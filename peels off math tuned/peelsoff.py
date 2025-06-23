import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

model_id = './tinyllama-lora-tuned/checkpoint-84'
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

while True:
    val = input('\nEnter Number (`q` or `e` to exist): ')
    if val == 'q' or val == 'e':
        break

    if val.isnumeric() == False:
        continue

    prompt = f'Peelsoff {val}'
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
