\# Expanded Attention Project

\### Author: Mike and Adam



\## Original Project

The original code for this was used last week for an attention model



\## Changes and Results

First, an option for quantization was added.  This resulted in an increase of 0.11% accuracy.  This is odd as quantization should train with smaller bit sizes and round the weights down for less precision.



Second, an option for sparse pattern was added.  This was tested with sparce levels of .1, .2 and .4 with the smaller level resulting in better accuracy.  This resulted in an increase of 0.09% accuracy from the baseline.



Finally, both options were tested together. This decreased the accuracy a bit but still 0.04% better than the baseline.









