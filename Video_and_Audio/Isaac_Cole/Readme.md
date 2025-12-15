# Encodec Audio Project
## Authors: Isaac Sperlich and Cole Drumheller

### Project Description
The goal of our project is to take an audio clip and manipulate it by messing around in latent space while using an Encodec model. We attempted to raise/lower the pitch of audio files using an encoder and decoder. We succeeded in changing the pitch, but we lose the words within the clips.

### Project Idea Selection
We initially wanted to do some things with Bark. Unfortunately there just wasn't much for us to tweak or work with inside of the Bark notebooks, so we went another route. We also looked into some sound effect generation with an Audio LDM Pipeline, however we ran into the same issues of not having much to really work with or manipulate. We like what we found using Encodec because we were able to do more with it, and effectively work with audio clips by using and messing with the latent space.