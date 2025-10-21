<img width="1536" height="226" alt="image" src="https://github.com/user-attachments/assets/bc37debe-66f5-4f93-a34d-01457da5476b" />

&#x205F;
&#x205F;

This project demonstrates the core mechanics of the Transformer architecture without relying on any largelanguage models (LLMs).
It implements Transformer encoder from the ground up using only PyTorch, applying it to the AG News dataset for text classification.
- Implement transformer encode architecture
- Trains completely from scratch
- Process data locally
- Evaluates model performance

This code is to prove that transformers are powerfull without pretraining and that self-attention and positional encoding alone can affectivly capture contextual relationships
in text and perform meaninful classification

This code recreates the essence of modern NLP architectures, showing how raw text can be transformed into high dimensional embeddings, passed through self-attention layers, and classified all without any pretrained LLMs.
It’s a proof-of-concept that bridges the gap between theory i.e "Transformer math" and practice "hands-on model training".


## Files
  - train.py
  - model.py
  - data.py
  - requirments.py
```
    pip install -r requirements.txt
```

## To Exectue 
```
python train.py --epochs 1 --d_model 64 --nlayers 1
```
## Output

<img width="1238" height="707" alt="image" src="https://github.com/user-attachments/assets/88e526ef-a27c-4019-9d2a-d3e9dad0def4" />

## Modifications
The only Hyper Parameters that made much difference were increasing the epochs or the number of heads.  We were able to get to 92% accuracy with 6 epochs and 16 heads.
Other modifications did not make much more accuracy.
