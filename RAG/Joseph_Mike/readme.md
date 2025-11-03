### Knowledge Graph RAG
For this project, we wanted to build a knowledge graph for our RAG system. This has proven to be very manual process that involves more database design than originally believed when conceptualized. When researching this problem it was found that there are unsupervised learning methods that can autonomously generate the graph structure for the model, but this is outside of our current understanding and might require more time than a weekend project.   

## main.py
This file implements a very basic, manually created knowledge graph. This implementation does not focus on adding retrieved documents to a prompt for analysis with a full power model but instead focuses on finding information in the graph through a cosign similarity with a small in memory vector database. Focusing on the node dataclass, in this project it contains simple key: value pairs of information that can be matched for similarity, in a larger project it could be imagined that this would instead contain larger documents or other information that can be integrated into a prompt to an LLM. If you were to experiment with the queries, you would notice they don't return very useful answers outside of the ones currently added; to improve this better integration with a large model would be required. 

## Output
The system runs several example queries to demonstrate how it retrieves relevant nodes and generates responses.

 Knowledge Graph RAG Proof of Concept
============================================================

Creating university knowledge graph...
Created KG with 14 nodes and 13 edges

Initializing encoder...
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
modules.json: 100% 349/349 [00:00<00:00, 30.7kB/s]config_sentence_transformers.json: 100% 116/116 [00:00<00:00, 7.70kB/s]README.md:  10.5k/? [00:00<00:00, 939kB/s]sentence_bert_config.json: 100% 53.0/53.0 [00:00<00:00, 5.14kB/s]config.json: 100% 612/612 [00:00<00:00, 59.1kB/s]model.safetensors: 100% 90.9M/90.9M [00:00<00:00, 153MB/s]tokenizer_config.json: 100% 350/350 [00:00<00:00, 26.3kB/s]vocab.txt:  232k/? [00:00<00:00, 10.5MB/s]tokenizer.json:  466k/? [00:00<00:00, 15.8MB/s]special_tokens_map.json: 100% 112/112 [00:00<00:00, 7.77kB/s]config.json: 100% 190/190 [00:00<00:00, 17.2kB/s]Loaded all-MiniLM-L6-v2 with dimension 384
Pre-computing node embeddings...
Computed embeddings for 14 nodes

============================================================
Running example queries...
============================================================

============================================================
Query: Computer Science major
============================================================

Retrieved 5 nodes:
  - Computer Science (major): 0.639
  - Mechanical Engineering (major): 0.461
  - Electrical Engineering (major): 0.444
  - Psychology (major): 0.409
  - Marketing (major): 0.383

Response:
Based on the university knowledge graph:

Computer Science is a major with 450 students.
It is offered by College of Engineering.

Related: Mechanical Engineering, Electrical Engineering.


============================================================
Query: engineering programs
============================================================

Retrieved 5 nodes:
  - Mechanical Engineering (major): 0.508
  - Electrical Engineering (major): 0.469
  - College of Engineering (college): 0.387
  - Computer Science (major): 0.379
  - Marketing (major): 0.310

Response:
Based on the university knowledge graph:

Mechanical Engineering is a major with 380 students.
It is offered by College of Engineering.

Related: Electrical Engineering, College of Engineering.


============================================================
Query: library hours
============================================================

Retrieved 5 nodes:
  - Main Library (facility): 0.603
  - Student Recreation Center (facility): 0.527
  - English Literature (major): 0.251
  - Computer Science (major): 0.228
  - School of Business (college): 0.226

Response:
Based on the university knowledge graph:

Main Library is open 24/7.
It has 5 floors.

Related: Student Recreation Center, English Literature.


============================================================
Query: School of Business dean
============================================================

Retrieved 5 nodes:
  - School of Business (college): 0.753
  - College of Liberal Arts (college): 0.594
  - College of Engineering (college): 0.586
  - Marketing (major): 0.436
  - Finance (major): 0.423

Response:
Based on the university knowledge graph:

School of Business is led by Dr. Jennifer Park.
This college offers 2 majors: Finance, Marketing.

Related: College of Liberal Arts, College of Engineering.


============================================================
Query: psychology students
============================================================

Retrieved 5 nodes:
  - Psychology (major): 0.636
  - Mechanical Engineering (major): 0.412
  - English Literature (major): 0.400
  - Electrical Engineering (major): 0.388
  - Computer Science (major): 0.385

Response:
Based on the university knowledge graph:

Psychology is a major with 520 students.
It is offered by College of Liberal Arts.

Related: Mechanical Engineering, English Literature.


============================================================
Query: Largest College
============================================================

Retrieved 5 nodes:
  - College of Engineering (college): 0.413
  - College of Liberal Arts (college): 0.409
  - School of Business (college): 0.402
  - Tech State University (university): 0.375
  - North Campus Housing (facility): 0.349

Response:
Based on the university knowledge graph:

College of Engineering is led by Dr. Sarah Chen.
This college offers 3 majors: Electrical Engineering, Computer Science, Mechanical Engineering.

Related: College of Liberal Arts, School of Business.

