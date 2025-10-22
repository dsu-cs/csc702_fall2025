# Image Classification with Transformers and Visualizing Attention
### Authors: Cole Drumheller and Joseph Doty

## Goal
The goal of this poroject was to trian a transformer to classify images. In addition to the classification we wanted to track the attention weights so that we could see a visual map of the attention for each image.

## Experiments and Results
### Patches
##### 4x4 Patches - 64 tokens
- With 20 epochs
    - Validation Accuracy 73-74%
    - Training time: 10 minutes
- With 600 epochs
    - Validation Accuracy leveled off around 84% at epoch 100
##### 2x2 Patches - 256 tokens
- With 20 epochs
    - Validation Accuracy 72-73%
    - Training Time: 20 minutes (Double 4x4)
- With 120 epochs
    - Validation Accuracy leveled off around 81% around epochs 90-100
##### 8x8 Patches - 16 tokens
- With 20 epochs
    - Validation Accuracy 65%
    - Training Time: 10 minutes
        - I expected this to be shorter than the 4x4, ended up being about the same
- With 40 epochs
    - Validation Accuracy seemed to be leveling around 72%

### Visualizing Attention
- Created an attention map to overlay over the original image which essentially shows a heatmap of the attention on the image
- Some images seemed to work better than others, but most images the main focus of the image was highlighted by the attention map
- Red on the heat map indicated higher attention, blue indicated lower

### Hyperparameter Tuning
- Tried to use Optuna to fine tune hyperparameters, didn't yield a better model