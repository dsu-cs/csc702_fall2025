
# Expanded attention

For this project i decided to use the same corpus as the previous assignment. The main thing I did different was try to add as many optimizations to the model as possible.

## Training Optimization

I used mixed precision to use 16 bit floats instead of 32 bits during the calculations. This reduces the memory required by half. It still keeps the 32-bit for weight updates so accuracy isn't sacrificed. I added in Gradient Accumulation, because large batches require lots of memory. This method splits the batches into smaller batches, processes them, and adds up their gradients. this reduces the required memory while maintaining the quality of larger batch sizes. Gradient clipping is also used to prevent weight updates from being too large or extreme. It checks to see if the gradient magnitude is greater than 1.0, if it is it scales it down to a max of 1.0. I also used a learning rate warmup and decay. The learning rate starts small and gradually increases, to prevent chaos and allow gentler weight updates. The decay drops the learning rate later in training, because most of the learning happens earlier. This allows for more fine tuning later in the training process. Lastly i used the AdamW Optimizer, which adjusts the learning rate per parameter based on past gradients. This helps prevent overfitting. The result of these optimizations is faster training, and smoother, more stable learning.

## Inference Optimization

Top-K Sampling: Keeps top 50 highest probabilities, zeros out the rest. Makes text more coherent.
Top-p Sampling: Keeps adding token until total probability is above 90 percent.
Example 1: "The capital of France is ___" → "Paris" (95% confident)
Only picks from 1-2 tokens
Example 2: "I feel ____  " → many emotions possible
Picks from 20+ tokens
Sequence Length Limiting: Essentially a sliding window for context. Always looks at the most recent context. Older tokens are "forgotten" but still influenced the recent ones

## Model Optimization

I used GELU activication instead of ReLU which is what chatGPT uses. This results in better gradients and better training. I also normalized before attention and the feedfoward network, because it has the potential to train a deeper model. I implemented 8 bit quantization which means the model stores weights as integers instead of floats, which results in a 4 times reduction in model size with a minimal accuracy loss
