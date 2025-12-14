
# Video and Audio project
## By Mike Kessler


The goal of my project was to take an audio recording, change it's style in some way, and then create a video of a something reacting to the new audio.  I was looking for something with emotion in it when I realized that my daughter is in Oral Interp in high school and has been working on a speech.   Her speech is based on the book "Are You There God.  It's Me, Margaret."

I had her record her speech and used that as the basis for the project.  The idea was that Whisper would detect the words she was saying and with which emotion.   Then each line is converted to it's opposite emotion and written out as an audio file with Bark.  The new audio file is then used as a basis for a video.  

This project did fall short of my goal of having a character use it's mouth to speak the speech, but not for a lack of trying.  I had many issues getting Whisper and Bark to both work as they had issues with current Torch libraries.   I had to use older libraries to make both sides stable.

Overally I did have success as the speech was transcribed and emotion was flipped.   I did get an audio file, but it is a bit rough in some areas.   It did contain speech with the new emotion settings.  The video uses the audio and shows a character that enlarges as the speaker talks.

The files involved:
  

 - MKesslerAudioVideo.ipynb  - The Python Notebook with the project
 - 20251207_174512.m4a - The initial audio of my daughter's speech
 - flipped emotions.mp3 - The converted output file
 
 The output video was too large to add to GitHub as it has a 25mb limit.
 

