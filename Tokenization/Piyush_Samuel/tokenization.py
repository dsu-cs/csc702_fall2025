from bpetokenizer import BPETokenizer
import wikipedia

# Pull wiki page for South Dakota and store content in a string
sd = wikipedia.page("South Dakota")
texts = sd.content

sd_tokenizer = BPETokenizer()
sd_tokenizer.train(texts, 500, verbose=True)

print(sd_tokenizer.vocab)

# Pull wiki page for North Dakota and store content in a string
nd = wikipedia.page("North Dakota")
texts = nd.content

nd_tokenizer = BPETokenizer()
nd_tokenizer.train(texts, 500, verbose=True)

print(nd_tokenizer.vocab)

# Print 10 longest tokens from each dataset
sd_longest_tokens = sorted(sd_tokenizer.vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]
nd_longest_tokens = sorted(nd_tokenizer.vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]

print(sd_longest_tokens)
print(nd_longest_tokens)