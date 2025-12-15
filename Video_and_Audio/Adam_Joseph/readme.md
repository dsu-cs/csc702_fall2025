Project Description:

    For this project, we settle on attempting to build a voice assistant. The tech stacks includes OpenAIs whisper library (ran locally, cpu), google's agent development kit (adk) (Requires api key sent via email to Jared, Subject is CSC 702 Joseph/Adam API Key), and some custom audio code for input. The general flow of the program is as follows: 
    
    1. The app is started.

    2. A volume level check is complete. This is a brief two seconds where the command line prompts to be quiet for a moment. 

    3. The app begins listening, it should detect when a user begins speaking, and start processing their request when they finish speaking. This is not a very complicated voice analysis tool (VAT), and we realized that could have been an iteresting project all on it's own. To stop the app simply say "Quit" when the app is listening or use a keyboard interupt. 

    4. The "agent" can only handle simple mock functions attibuted to the adk introduction page: https://google.github.io/adk-docs/get-started/quickstart/#agentpy. After speaking to the agent it will tell you what it can do or that it cannot help with your request, utilize one of these tools, or quit. You can ask through your voice for the weather in "new york" (Not "New york city", that was enough to break the function at times). And the same for the time in "new york," any other cities the provided function just says "I cannot assist with that." 

    5. The listening loop is set to stop after 5 iterations. 

Setup: 

    I used a python virtual environment locally but that might not be required. 

    1. pip install -r requirements.txt (If this doesn't work we are expecting: pip install numpy, torch, openai-whisper, sounddevice, google-adk)

    2. Create a .env file in the same folder as main.py, the information to save here will be provided in the email.

    2. Run main.py (I just did this in vscode using the run without debugging in menu)

    3. The app should start the loop described above.


Issues: 

    1. The listening fuctionality can be unreliable. There are settings to adjust it's thresholds, but it is best tested in a relatively quiet room. We are unsure how microphone quality will affect it. 

    2. Your editor might flag main.py as having some method errors. The type hinting is saying that a list[] is expected when in reality a str type should be sent if the speech to text functionality works correctly. In testing this worked perfectly fine at runtime. 

    3. Exception handling is lacking. Assuming the audio is recorded correctly, the api doesn't encounter a 429 resource exhaustion error(Seems like many people are encountering this error with gemini currently even if they are paid), the program should work as described above. 



Room for improvement: 

    1. We attempted to persist user sessions or conversations, but that provided some trouble and isn't in the final project. 
