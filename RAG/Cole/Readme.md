# RAG with Class Notes
### Author: Cole Drumheller

## Description/Goal
The goal of this project was to utilize RAG to allow chatting with an OpenAI LLM about class topics. The LLM also cites the documents it uses to answer the question. I uploaded both lecture notes and note-taking frameworks from the class.

## Experiments
#### Chunk Size and Documents Retrieved
- Default Values
    - Chunk Size = 1000
    - Documents Retrieved = 3
    - This worked well, it typically got 2 documents from the note-taking framework and then it would pull 1 document that was lecture notes.
- Other Combos Tried
    - Chunk Size = 500, Documents Retrieved = 2
        - This seemed to not capture enough information, it missed some things. It was still ok, but not as good as the 1000 window.
    - Chunk Size = 1000, Document Retrieved = 2
        - This worked a lot better than the previous, I think the 500 character window was just too small when looking at the note-taking summaries. This was pretty comparable to the default values. Typically only pulled documents from note-taking frameworks.
    - Chunk size = 1500, Documents Retrieved = 2
        - This also worked well, once again the 2 documents retrieved were usually note-taking frameworks. Maybe a little bit better than the 1000 chunk size.

## Analysis
Overall, I think the more important of the 2 hyperparameters is the chunk size. Document size is important too, but for a smaller application such as this it's more important to get enough context within each chunk. I think this is a cool little tool, and honestly I'll probably use this as the year goes on for a couple different classes. Either for studying, finding where I made notes about certain topics, or making it easier to analyze academic papers.
