from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

model_id = './results/tinyllama-lora-tuned/checkpoint-84'
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def generate(num:int):
    prompt = f'Peelsoff {num}'
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
    return response

print(generate(33))
print("-" * 33)

while True:
    val = input('\nEnter Number (`q` or `e` to exist): ')
    if val == 'q' or val == 'e':
        break

    if val.isnumeric() == False:
        continue

    print(generate(int(val)))

    
