import pandas as pd
import random
import numpy as np

nums = [0, 1, 4, 5, 8, 10, 100, 1001, 500,
        1000, 1001, 7777, 12345, 123456, 1234567]

# random_integers = [random.randint(1, 999999) for _ in range(100)]
def generate_nums(size:int, min: int, max: int):
    random_nums = [random.randint(min, max) for _ in range(size)]
    return random_nums

nums.extend(generate_nums(15, 101, 1000))
nums.extend(generate_nums(15, 1001, 10000))
nums.extend(generate_nums(10, 10001, 100000))
nums.extend(generate_nums(10, 100001, 1000000))
nums.extend(generate_nums(10, 1000001, 10000000))
nums.extend(generate_nums(10, 10000001, 100000000))
nums.extend(generate_nums(10, 100000001, 1000000000))
nums.extend(generate_nums(10, 1000000001, 10000000000))

nums = sorted(nums, reverse=False) #sort array
nums = list(dict.fromkeys(nums)) #remove duplicate value

def sum_of(num: int):
    total_sum: int = 0
    temp = abs(num)
    nums = []
    while temp > 0:
        cur = temp % 10
        nums.append(str(cur))
        total_sum += cur
        temp = int(temp / 10)

    sol = " + ".join(nums)
    if len(sol) == 0:
        sol = '0'
    text = f'Sum of digits: {sol[::-1]} = {total_sum} \nAnswer: {total_sum}'
    return [int(total_sum), text]

def create_question(num:int) -> str:
    return f'Peelsoff {num}'
def create_response(num:int) -> str:
    d = sum_of(num)
    return f'{d[1]}'

df = pd.DataFrame()
df['number'] = nums
df['question'] = df['number'].apply(create_question)
df['answer'] = df['number'].apply(create_response)
df = df.drop(columns=['number'])
df.to_csv('data/train_data.csv', columns=['question', 'answer'])

print('Successfully training data created')
