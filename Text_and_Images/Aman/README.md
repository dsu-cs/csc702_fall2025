# Text-Triggered Image Modification

This project shows how an image can be modified using a text description.
The user uploads an image, writes a prompt, and the system generates a modified version of the image.

The project uses diffusion models to perform image-to-image editing.

---

## What the Project Does

* Takes an image as input
* Takes a text prompt from the user
* Modifies the image based on the prompt
* Displays the output in a web interface

The main focus is on changing the background or visual style while keeping the subject structure.

---

## How It Works

1. The user uploads an image
2. The image is resized to a valid size
3. A depth map is created to preserve structure
4. A diffusion model edits the image using the text prompt
5. The output image is shown to the user

---

## Tools Used

* Python
* Stable Diffusion (img2img)
* ControlNet (depth guidance)
* Hugging Face Diffusers
* Gradio
* Google Colab

---

## How to Run

1. Open Google Colab and enable GPU
2. Install required libraries
3. Run the notebook
4. Upload an image and enter a prompt

---

## Example Prompt

```
professional studio photography, deep black background, dramatic lighting
```

Negative prompt example:

```
snow, white background, blurry, low quality
```

---

## Notes

* Results depend on the prompt quality
* Background replacement may not always be perfect
* ControlNet improves structure but slows generation
