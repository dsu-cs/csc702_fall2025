Usually, a stable diffusion model will take a fully noisy image and remove fractions of noise one  (or fraction) at a time to determine the most likely variation of the initial image prompt. From this, we propose testing the theory that if we remove higher fractions of noise at a time until it reaches the idea of the initial image instead of just one step of noise at a time, it will be quicker but probably more dissimilar to the initial image prompts actual meaning. 

We will analyze the final images (of the outputted idea of the initial image) from these two variations of the stable diffusion model. These will form a comparison against the actual initial image prompt, and we will hopefully be able to determine some statistical and numerical observations from these images determining accuracy/efficiency/timing. Can compare CLIP similarity between prompt and Images or MSE if using a base image.

We also would like to implement displaying example images of the denoising process and see the image as it transforms. We also are planning to implement a version where the pace at which it removes noise will dynamically change depending on where in the process it is (e.g. faster at the start and slower at the end or something like this). We also should try differrent scheduler types (DDPM, DDIM, DPMSolverMultistep, Euler, etc.)


Upon doing some research, we discovered we need to actually adjust the rate of total timesteps for the scheduler to adjust how much noise is removed. So now the actual code goes into variations of how to do that (as described above for the different variation proposals). 

Additionally:
 
    Baseline is the base with a standard amount of timesteps.
    Coarse is essentially using fewer timesteps and running through it (giving a coarse and potentially less accurate image)
    Dynamic is using a dynamic scheduler that can have its pacing alternate (adjusting how much noise is removed at different times in the process).

CLIP measures similarity between prompts and images. LPIPS compares images to a potentially provided ground-truth image. Time is also measured as this is also important.



In the .ipynb file we install, import, and configure our necessary information first. Then we move to build the scheduler based on the given parameter, and move to initialize and load the pipeline information. We then load in our metrics.

Then we go on to make custom timesteps for the scheduler and implement this if desired.

All of this was done as a more modular process and certain parts can not be done or be done. They can also be adjusted for things like scheduler type or seed or prompt or etc...

Then we run our experiment. This ended up giving us our results. From here we display all of our resulting images along the duration of one run in a comparitive fashion (you can easily examine multiple types like baseline vs. coarse vs. dynamic all side to side). We also end by finally displaying our final images and having a brief analysis.



ANALYSIS: 
The results from the 3 different timestep variations were not what we expected. All of the final images appeared to be identical to the human eye, and even with the numeric scoring using CLIP and LPIPS we see very little difference between the 3 images.  clip_baseline  clip_coarse  clip_dynamic  
0       0.365404     0.375605      0.365404  
1       0.356073     0.368277      0.356073 

The running time did improve significantly from the baseline method, with a roughly 15 second decrease in running time for the coarse method. 
prompt_idx  seed  time_baseline_s  time_coarse_s  time_dynamic_s  \
0           0    23        36.579671      20.558656       34.017078   
1           0    42        34.906709      19.937115       34.531759   

The uneventful results could be a result of many different factors. The model we are using is very well trained and that could be compensating for the lack of timesteps. 