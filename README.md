# DocuChat: Question Answering Chatbot using RAG System
A lightweight Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions in natural language. Instead of relying solely on the knowledge of a Large Language Model (LLM), the system retrieves relevant information from the uploaded document and uses it to generate accurate, context-aware responses.

Beyond building the application, this project investigates how different document chunking strategies and embedding models influence retrieval accuracy and response latency in RAG systems.


## Inspiration
The KDSH 2026 problem statement motivated us to explore one of the key challenges in modern Large Language Models: reasoning over long contexts. The challenge demonstrated that although LLMs excel at localized understanding, they often struggle to maintain consistency and effectively aggregate information across lengthy documents.

To address this limitation, we turned to Retrieval-Augmented Generation (RAG), a framework that enhances LLMs by retrieving the most relevant information from external documents before generating responses. Since the effectiveness of a RAG system heavily depends on how documents are represented and retrieved, our work focuses on evaluating the impact of chunk size and embedding model selection on retrieval accuracy, response quality, and system efficiency.


## Objectives
We focused mainly on two objectives: 
1. Develop a simple and efficient PDF Question Answering system using RAG.
2. Experimentally evaluate how different chunk sizes and embedding models influence retrieval performance.

Specifically, we investigate:
- Does chunk size affect retrieval quality?
- Which embedding model provides better semantic search?
- What is the trade-off between response quality and response time?


## System Overview

```
                    PDF
                     │
                     ▼
            Text Extraction
                     │
                     ▼
            Document Chunking
                     │
                     ▼
          Generate Embeddings
                     │
                     ▼
              Chroma Vector DB

────────────────────────────────────

              User Question
                     │
                     ▼
           Embed User Query
                     │
                     ▼
          Similarity Search
                     │
                     ▼
       Retrieve Relevant Chunks
                     │
                     ▼
       Prompt + Retrieved Context
                     │
                     ▼
          Large Language Model
                     │
                     ▼
                 Response
```


## Experimental Setup

To understand how different retrieval configurations affect performance, we compare multiple combinations of chunk sizes and embedding models.

### Chunk Sizes
256 Tokens   
512 Tokens

### Embedding Models
all-MiniLM-L6-v2   
BAAI/age-small-en  


## Evaluation Metrics
The following metrics are used to compare different configurations.


### Retrieval Accuracy
Measures whether the retrieved document chunks contain the information necessary to answer the user's question.


### Response Latency
Measures the total time taken to retrieve relevant information and generate the final response.


### Answer Quality
Responses are evaluated based on:
-Correctness
-Relevance
-Completeness


## Dataset Used

### College Examination Timetable
Example Questions-  
-Which exam does Roll No. 2205143 have on 18 December?
-When is the Operating Systems examination?
-Which room is allotted for Roll No. 2206125?
