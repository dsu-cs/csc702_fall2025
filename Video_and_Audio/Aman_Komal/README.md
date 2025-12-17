# Automatic Sound Generation from Silent Videos

## Overview

This project generates sound effects automatically from silent videos.

The system looks at the motion in a video, finds important action moments (like punches or hits), and adds sound at the correct time.
No manual sound editing is needed.

The main goal of this project is **good timing and alignment**, not perfect studio-quality sound.

---

## What problem does this solve?

Many videos:

- have no sound
- have bad or missing sound effects

Adding sound manually:

- takes a lot of time
- needs editing skills

This project automates the process by:

- analyzing video motion
- deciding _when_ sound should happen
- generating sound automatically

---

## Dataset

We used the **UCF-101** action recognition dataset.

From this dataset, we focused on actions like:

- Punch
- Cricket Shot

Important:

- Videos already had sound
- We **removed the original audio**
- Our system works only on **silent videos**

---

## How the system works (high level)

1. Take a silent video
2. Extract motion information using optical flow
3. Create a motion graph over time
4. Detect strong motion moments (impacts)
5. Generate sound at those moments
6. Merge the generated sound with the video

---

## Step 1: Motion extraction

- The video is split into frames
- Optical flow is used to measure movement between frames
- Each frame gets a **motion value**
- This creates a motion signal over time

This motion signal tells us _where something important happened_.

---

## Step 2: Motion smoothing and impact detection

Raw motion is noisy.

So we:

- smoothed the motion signal using a sliding window average
- grouped motion into bursts
- selected only the strongest moment from each burst

This helped:

- reduce false sounds
- improve timing

---

## Step 3: Procedural sound generation

The first approach was **procedural audio**.

How it works:

- Create a silent audio track
- Generate short noise bursts
- Apply fast decay (loud → quiet)
- Place the sound at detected impact times

This gave:

- good timing
- simple but artificial sound

---

## Problems we faced

1. Sound not aligned properly at first
2. Too many sounds for one action
3. Sound felt unnatural
4. Small timing errors were noticeable

---

## Fixes we applied

- Better motion smoothing
- Burst-based impact detection
- Only one sound per action
- Small timing shift (30–40 ms earlier) for natural feel

These fixes improved alignment a lot.

---

## Diffusion model for better sound

To improve sound quality, we added a diffusion model.

**Model used:**

- `teticio/audio-diffusion-256`

Why this model:

- Pretrained
- Easy to use
- Works on CPU
- Produces realistic impact sounds

How we used it:

- Motion detection still decides _when_
- Diffusion model decides _how the sound sounds_
- Generate one impact sound
- Place it at the detected moment

---

## Procedural vs Diffusion audio

| Method     | Pros                       | Cons              |
| ---------- | -------------------------- | ----------------- |
| Procedural | Fast, simple, controllable | Sound is basic    |
| Diffusion  | Realistic sound            | Slow, heavy model |

Best result:

- Procedural method for timing
- Diffusion model for sound quality

---

## Audio and video merging

We used **FFmpeg** to merge audio and video.

Important:

- FFmpeg does **not** align sound
- Alignment is done in code
- FFmpeg only combines audio and video timelines

---

## Results

- Motion-based sound placement works
- Alignment improved step by step
- Diffusion model gives better sound quality
- Some videos are perfectly aligned
- Overall result is much better than manual rules

---

## Limitations

- Diffusion model is slow
- Not real-time
- Works best for short impact sounds
- Complex scenes may still confuse motion detection

---

## Conclusion

This project shows that:

- Motion alone can guide sound placement
- Good timing matters more than perfect sound
- Simple methods can work well
- Diffusion models improve realism when used carefully

---

## Tools and libraries used

- Python
- OpenCV
- NumPy
- PyTorch
- FFmpeg
- UCF-101 dataset
- Audio Diffusion model
