# Text Summarization with Seq2Seq + Attention and BART

This project demonstrates abstractive text summarization on the *CNN/DailyMail* dataset using two approaches:

1. *Custom Seq2Seq + Attention model (PyTorch)*  
   - Trains a GRU-based encoder–decoder with attention.  
   - Tokenizer built using SentencePiece (BPE).  
   - Lightweight, educational model (not production-ready).  

2. *Pretrained BART (facebook/bart-large-cnn)*  
   - State of the art transformer model.  
   - Produces high-quality summaries.  

---

## Setup

Install dependencies:

```bash
pip install datasets sentencepiece tqdm transformers
```

## Run the script:

python summarization.py

It will:

Train the custom seq2seq model for 5 epochs on a small subset of CNN/DailyMail.

Save checkpoints (seq2seq_ckpt.pt).

Generate summaries from both the custom model and pretrained BART for comparison.


## Notes

The custom model requires longer training (20–50 epochs) to produce more fluent summaries.

The pretrained BART model works immediately and is recommended for real-world use.