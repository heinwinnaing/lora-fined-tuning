# Peels Off (calculates the sum of the digits)

The function effectively "peels off" the last digit of the number repeatedly using the modulo operator and adds it to a running sum, then removes that digit using integer division, until the number becomes zero.

Peelsoff 123
Sum of digits: 1 + 2 + 3 = 6

---

# 1. Create training data

```
python create_data.py
```

Successfully training data created

---

# 2. Training model
```
python fine-tuning.py
```

Training started...
{'loss': 4.7884, 'grad_norm': 11.431668281555176, 'learning_rate': 1.928571428571429e-05, 'epoch': 0.19}
{'loss': 0.0905, 'grad_norm': 2.642010450363159, 'learning_rate': 1.8333333333333333e-05, 'epoch': 0.38}
{'loss': 0.0806, 'grad_norm': 3.119539976119995, 'learning_rate': 1.7380952380952384e-05, 'epoch': 0.57}
{'loss': 0.0908, 'grad_norm': 1.4752193689346313, 'learning_rate': 1.642857142857143e-05, 'epoch': 0.76}
{'loss': 0.0702, 'grad_norm': 2.395752191543579, 'learning_rate': 1.5476190476190476e-05, 'epoch': 0.95}
{'train_runtime': 1479.057, 'train_samples_per_second': 0.224, 'train_steps_per_second': 0.057, 'train_loss': 0.28559545392081853, 'epoch': 4.0}
100%|█████████████████████████████████████████████████████████████████████████████████| 84/84 [24:39<00:00, 17.61s/it]
Training completed

---
# 3. Tesing the fine-tuned model
```
python peelsoff.py
```

Peelsoff 33
Sum of digits: 3 + 3 = 6
Answer: 6
#---------------------------------

Enter Number (`q` or `e` to exist):
