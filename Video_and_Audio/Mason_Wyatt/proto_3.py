import torch
from diffusers import AutoencoderKLCogVideoX
from diffusers.utils import export_to_video
import gc
import numpy as np

torch.cuda.empty_cache()
gc.collect()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load 3d vae 
# use float16 to save memory
model_id = "THUDM/CogVideoX-2b"
vae = AutoencoderKLCogVideoX.from_pretrained(
    model_id,
    subfolder='vae',
    torch_dtype=torch.float16,
).to(device)

# enables tiling which should save on vram usage
vae.enable_tiling()

# helper to prep video for vae
# need to replace this function for better use later
def load_and_process_video(video_path, height=480, width=720, frames=8):
    fake_video = torch.rand(1,3,frames,height,width).to(device, dtype=torch.float16)
    fake_video = (fake_video * 2.0) - 1
    return fake_video

# manually splits vid into temporal chunks to save vram
def get_latents_chunked(video_tensor, chunk_size=4):
    frames = video_tensor.shape[2]
    latent_list = []

    with torch.no_grad():
        for i in range(0, frames, chunk_size):
            # get slice of frame
            end = min(i + chunk_size, frames)
            video_chunk = video_tensor[:,:,i:end,:,:]

            # encode current chunk
            posterior = vae.encode(video_chunk).latent_dist
            latents = posterior.sample() * vae.config.scaling_factor
            latent_list.append(latents)

            # clean up vram
            del video_chunk, posterior, latents
            torch.cuda.empty_cache()

    # stitch chunks together and return
    return torch.cat(latent_list, dim=2)


# decode latents & manually split to save vram
def decode_latents_chunked(latents, chunk_size=1):
    latent_frames_count = latents.shape[2]
    decoded_video_list = []

    with torch.no_grad():
        for i in range(0, latent_frames_count, chunk_size):
            # slice latent
            end = min(i+chunk_size, latent_frames_count)
            latent_chunk = latents[:,:,i:end,:,:]

            # decode current chunk
            frames = vae.decode(latent_chunk).sample

            # move to cpu to free vram
            frames = (frames / 2 + 0.5).clamp(0,1)
            decoded_video_list.append(frames.cpu())

            # cleanup
            del latent_chunk, frames
            torch.cuda.empty_cache()
    # stitch together and return
    return torch.cat(decoded_video_list, dim=2)




# ---- TESTING ---- #
print('creating dummy vid')
video_a = load_and_process_video('video_a.mp4', frames=1024)
video_b = load_and_process_video('video_b.mp4', frames=1024)

print('extracting latents')
latents_a = get_latents_chunked(video_a)
latents_b = get_latents_chunked(video_b)

print(f'latent shape: {latents_a.shape}')

# manipulation
alpha = 0.5
latents_hybrid = (latents_a * (1-alpha)) + (latents_b * alpha)

print('decoding hybrid vid')
decoded_frames = decode_latents_chunked(latents_hybrid)

video_tensor = decoded_frames[0]
video_tensor = video_tensor.permute(1,2,3,0)
video_np = video_tensor.cpu().numpy()
video_np = (video_np * 255).astype(np.uint8)

print('saving vid')
export_to_video(video_np, 'output.mp4', fps=16)