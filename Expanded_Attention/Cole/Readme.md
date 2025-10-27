# Expanded Attention Project
### Author: Cole Drumheller

## Original Project
The original code for this was designed to train a transformer to do Image Classification. After training the code used MatPlotLib to show the images with attention heat map overlays.

## Changes to implement Expanded Attention
To add expanded attention to this transformer I added the get_attention_mask function. This function creates a window size for each token based on the layer of attention we are currently in. The window size increases as we go, so the smallest window is in the first layer and a global window is used in the last layer. I also updated the visualization code to show each layer in its own image.
- Changes to the original code are labeled by markdown cells above the modified code cells.

## Compared Performance
The expanded attention hurt the accuracy of the transformer. It dropped from the low 70s to the low 50s. It also made the training go slightly slower, 20 epochs originally took 10 minutes, it now takes just over 30 minutes.