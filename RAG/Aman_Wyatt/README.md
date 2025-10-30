# TBD RAG Application

## Issues we ran into
When first trying to set things up, we attempted to do everything locally on the vm. Our motivation was to use small models that were more likely limited in their knowledge in order to have more noticable results when incorporating extra information/context with RAG. We were using ollama and some of the infrastructure built around that along with Qdrant for our vector database that ran on a docker container. We ran into issues with memory usage, despite utilizing small models. Rather than beating our heads against the wall just trying to get our pipeline set up, we transferred over to utilizing API calls to Groq, mainly because it seemed to be the most straighforward and it didn't sit behind a paywall like several others.
