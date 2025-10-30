# TBD RAG Application

## Issues we ran into
When first trying to set things up, we attempted to do everything locally on the vm. Our motivation was to use small models that were more likely limited in their knowledge in order to have more noticable results when incorporating extra information/context with RAG. We were using ollama and some of the infrastructure built around that along with Qdrant for our vector database that ran on a docker container. We ran into issues with memory usage, despite utilizing small models. Rather than beating our heads against the wall just trying to get our pipeline set up, we transferred over to utilizing API calls to Groq, mainly because it seemed to be the most straighforward and it didn't sit behind a paywall like several others.

## Alternate approach
We pivoted to our Groq API calls and got a very basic llama 3.1 model running that really wanted to talk about RAG (it only had 2 "documents" which were each a phrase regarding RAG). We then transfered this into two programs that run interactively as chatbots. The `chatbot.py` script is simple and runs in the terminal as a normal python script. The `pretty_chatbot.py` file is run using `streatlit run pretty_chatbot.py` and provides an easy-to-use web interface to interact with the bot through.
**Be sure to insert your Groq API key before running**

## Required Packages
`sentence-transformers`
`chromadb`
`groq`
`streamlit`