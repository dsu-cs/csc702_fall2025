We steal a working transformer from Josh at https://github.com/StatQuest/decoder_transformer_from_scratch/blob/main/decoder_transformers_with_pytorch_and_lightning_v2.ipynb

After modifying the input to take in our epic of Dracula.txt and after making sure it works, we try to add the expansion technique of sparse attention.

I have to horribly cripple training because I don't want to spend hours waiting, so model performance doesn't seem that great... it does appear to work though!

Since testing efficacy of massive context windows doesn't seem very simple, especially under certain time constraints, I'll be aiming for a model that still appears to work after I implant some "improvements" into its architecture. 

I think I did achieve some form of sparse attention, it just probably doesn't make much of a difference at the low sequence lengths used for the model. 