# Generating Images Project
## Authors: Cole Drumheller and Mike Kessler
## Final Project File: FinalVersion.ipynb

### Summary
In our project we used a pretrained diffusion model to generate images. We experimented with different hyperparameters such as guidance_scale and steps, as well as seeing what impact adding a negative prompt would have. We tried to get the model to create an image of baseball player hitting a baseball, and it struggled mightily which led to the main focus of the project, playing with hyperparameters and the prompts to get it to generate a decent image.

### Experiments and Results
- Guidance Scale
    - The guidance_scale hyperparameter effects how closely the model follows the prompt, the default value is 7.5 for the model we used. We started with a lower range, 4 to 12 and then increased to 15 to 25. The 15 to 25 range gave better results when generating a baseball batter when using a negative prompt as well..
- Inference Steps
    - The number of inference steps is the denoising iterations that are done when generating the image. The higher number of steps, the more detail in the image. I expected the higher number of steps to create better images, but we tried with up to 700 steps and it didn't really improve the image.
- Negative Prompt
    - The negative prompt tells the model things it shouldn't be generating in the image. After the guidance_scale and steps didn't make the images much better we added a negative prompt with some things that were messing with some of the images. A big one was "glove", many images showed batters with gloves instead of bats. The addition of the negative prompt made for a better image.
- Prompt
    - In addition to adding a negative prompt, we also changed the original prompt to try and be simpler and more specific. The inital prompt was just a baseball player hitting a ball, which is pretty vague. The more specific prompt, along with the other tweaks helped make more consistent images, even if they weren't all perfect.
