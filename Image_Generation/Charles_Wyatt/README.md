# Generating Images Project
### Wyatt & Charles

## The idea:
We set out to train a UNet model such that it removes noise that might appear in semi-corrupted images every-day people would come across. It learns to remove salt & peppering, small guassian-like blurs, compression noise and some horizontal blurring. These corrections are done in just one pass through the model. We hoped to make a model that had the potential be adapted into a tool that everyday people could use to increase the quality of any of their photos.

## Our data:
We use the [Flickr-Faces-HQ Dataset (Nvidia) - Resized 256px](https://www.kaggle.com/datasets/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px) for our images. This provides us with 70,000 256x256 images to utilize. We generate our input data synthetically by randomly adding the noising techniques mentioned above and use this dataset as our ground truth for the model to compare its outputs against. This dataset is not included in our github submission

## Files overview:
A quick rundown of what to look for in each file.
* `demo.ipynb` - Visualization of results with example(s)
* `train.ipynb` - Hyper-parameter tuning and model training code
* `split_flickr.py` - Data prep for splitting our data into validation and training sets
* `utils.py` - Early utilities such as our custom dataloader and UNet model

## Results and remarks
