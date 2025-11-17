# Generating Images Project
### Wyatt & Charles

## The idea:
We set out to train a UNet model such that it removes noise that might appear in semi-corrupted images every-day people would come across. It learns to remove salt & peppering, small guassian-like blurs, compression noise and some horizontal blurring. These corrections are done in just one pass through the model. We hoped to make a model that had the potential be adapted into a tool that everyday people could use to increase the quality of any of their photos.

## Our data:
We use the [Flickr-Faces-HQ Dataset (Nvidia) - Resized 256px](https://www.kaggle.com/datasets/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px) for our images. This provides us with 70,000 256x256 images to utilize. We generate our input data synthetically by randomly adding the noising techniques mentioned above and use this dataset as our ground truth for the model to compare its outputs against. This dataset is not included in our github submission

## Files overview:
A quick rundown of what to look for in each file.
* `demo.ipynb` - Visualization of results with example(s)
* `train.ipynb` - Hyper-parameter tuning and model training code (also with PSNR and SSIM evaluation)
* `split_flickr.py` - Data prep for splitting our data into validation and training sets
* `utils.py` - Early utilities such as our custom dataloader and UNet model

## Results and remarks

We completed a full training run with the UNet model (20 epochs and 3500 batches per epoch), resulting in a training loss of 0.0004.

We evaluated the performance on the full training dataset using peak signal to Noise Ratio (PSNR): the range 30 - 40db indicates a strong denoising and reconstruction performance. Structural similarity index(SSIM): the  range of 0.87 - 0.97 shows that the model preserves important structural details from the clean image 

At the end of the evaluation the model achieved the following average performance:
* AVG PSNR: 34.602 db
* AVG SSIM: 0.9052

**Our model is not uploaded to github because the file is too large!**
The model seems to do decent with removing the noise added to the images; however, there are occasionally some quirks that show through, and the model seems to be more likely to trip up when multiple noise methods are added to an image. This may very well be because it sees relatively few of these in training.