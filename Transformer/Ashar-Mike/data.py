# data.py
#Ashar And Mike
# This module handles the file loading and tokenization

#TorchText version throwing error on windows environment that is why used CSV data.
"""
TorchText-free data loader for AG News classification.
Downloads CSVs and builds a tiny vocab/tokenizer from scratch.
"""
from typing import List, Tuple
import os, csv, re, urllib.request
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
_TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
_TEST_URL  = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
_TRAIN_CSV = os.path.join(_DATA_DIR, "ag_news_train.csv")
_TEST_CSV  = os.path.join(_DATA_DIR, "ag_news_test.csv")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\w\s]")

def _basic_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]

def _ensure_data():
    os.makedirs(_DATA_DIR, exist_ok=True)
    if not os.path.exists(_TRAIN_CSV):
        urllib.request.urlretrieve(_TRAIN_URL, _TRAIN_CSV)
    if not os.path.exists(_TEST_CSV):
        urllib.request.urlretrieve(_TEST_URL, _TEST_CSV)

def _read_csv(path: str) -> List[Tuple[int, str]]:
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        for label, title, desc in csv.reader(f):
            text=f"{title}. {desc}"
            rows.append((int(label), text))
    return rows

class _Vocab:
    def __init__(self, stoi): self._stoi=stoi
    def __len__(self): return len(self._stoi)
    def __call__(self,tokens): return [self._stoi.get(t,self._stoi["<unk>"]) for t in tokens]
    def get_stoi(self): return self._stoi

def _build_vocab(train, min_freq=2):
    from collections import Counter
    c=Counter()
    for _,text in train: c.update(_basic_tokenize(text))
    stoi={"<unk>":0,"<pad>":1}
    for tok,freq in c.items():
        if freq>=min_freq: stoi[tok]=len(stoi)
    return _Vocab(stoi), stoi["<pad>"]

def get_dataloaders(batch_size, device, min_freq=2):
    _ensure_data()
    train=_read_csv(_TRAIN_CSV)
    test=_read_csv(_TEST_CSV)
    vocab,pad_idx=_build_vocab(train,min_freq)

    def text_to_ids(t): return vocab(_basic_tokenize(t))
    def collate(batch):
        labels,seqs=[],[]
        for label,text in batch:
            labels.append(label-1)
            seqs.append(torch.tensor(text_to_ids(text)))
        padded=pad_sequence(seqs,batch_first=True,padding_value=pad_idx)
        return padded.to(device),torch.tensor(labels,device=device)
    return (
        DataLoader(train,batch_size=batch_size,shuffle=True,collate_fn=collate),
        DataLoader(test,batch_size=batch_size,shuffle=False,collate_fn=collate),
        vocab,pad_idx
    )
