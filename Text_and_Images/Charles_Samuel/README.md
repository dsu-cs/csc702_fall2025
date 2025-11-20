# Linking Images to Text Project - Comparing CLIP results with diffusion generated images
For this project, we used pretrained CLIP and diffusion models to generate images based on the most similar caption to a given image. Our original idea for the project was to do a reverse prompt search, where you enter an image into the CLIP model and, out of a list of possible captions, it will give you the most similar. We completed this reverse prompt search quite easily, so we decided to add more to it. We decided to take the closet caption to the given image and use it as a prompt for a diffusion model, then compare the original input image and the generated image. To compare the two images, we first encoded both into latent embeddings so we could use cosign similarity.

We did this process of generating an image off the closet caption and comparing results with multiple different input images. For the first few trials, we wanted to pick out images that directly correlated with one of the caption in our list. The first two images returned with pretty good scores, showing the transfer from CLIP caption to diffusion generation did not loss too much information. But with the third image, the CLIP model was able to correctly identify "a bird playing guitar", but the diffusion model struggled to create a good image. With the last couple tests, we inputted images that did not have a directly related caption in our list. We inputted an image of sharks playing pool and it ended up creating an alligator with a top hat. We inputted a moose in a street and it outputted a deer in a city street. Even though there were no correct captions, the CLIP model was still able to pull specific details from the image to help generate a somewhat similar image, like the sharks wearing top hats and the moose standing in a city street.

## Steps
1. Uses CLIP to match an input image with a generated caption.
2. Uses Stable Diffusion to generate a new image from that caption.
3. Uses a different CLIP model to extract embeddings.
4. Computes cosine similarity between original and generated images.
5. Outputs original image, similarity score, and generated output.

## Requirements
- torch
- transformers
- diffusers
- pillow

## How to Run
Open the notebook and run all cells in order.
