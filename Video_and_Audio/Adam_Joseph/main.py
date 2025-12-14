import numpy
import torch
import whisper
import sounddevice as sd

# Configurations
SAMPLE_RATE = 16000
DURATION = 5
CHANNELS = 1

# Select GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup speech recognition model
model = whisper.load_model("turbo", device=device)

# Capture audio
print("Recording for 5 seconds...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
sd.wait()
print("Done recording.")
print(f"Audio shape: {audio.shape}")

# Convert audio into a format that whisper expects, a 1D array. 
np_audio = audio.squeeze().astype(numpy.float32)

# Pass the audio to the model
response = model.transcribe(np_audio, fp16=False)

# Check the audio
print("Transcribed test: ", response["text"])