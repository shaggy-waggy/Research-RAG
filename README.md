# DocuChat: PDF Question Answering Chatbot with RAG

DocuChat is a lightweight Retrieval-Augmented Generation (RAG) system that lets users upload a PDF and ask questions in natural language. Instead of relying only on the knowledge of a large language model, the system retrieves relevant chunks from the uploaded document and uses them to generate grounded, context-aware answers.

The project also serves as a research prototype for comparing retrieval configurations such as chunk size, embedding model, and retriever type.

## Inspiration

The KDSH 2026 problem statement motivated us to explore one of the key challenges in modern Large Language Models: reasoning over long contexts. The challenge demonstrated that although LLMs excel at localized understanding, they often struggle to maintain consistency and effectively aggregate information across lengthy documents.

To address this limitation, we turned to Retrieval-Augmented Generation (RAG), a framework that enhances LLMs by retrieving the most relevant information from external documents before generating responses. Since the effectiveness of a RAG system heavily depends on how documents are represented and retrieved, our work focuses on evaluating the impact of chunk size and embedding model selection on retrieval accuracy, response quality, and system efficiency.

## Objectives
We focused mainly on two objectives: 
1. Develop a simple and efficient PDF Question Answering system using RAG.
2. Experimentally evaluate how different retrieval types, chunk sizes and embedding models influence retrieval performance.

Specifically, we investigate:
- Does chunk size affect retrieval quality?
- Which embedding model provides better semantic search?
- What is the trade-off between response quality and response time?

## What the app does

- Upload a PDF in the Streamlit app
- Process and index the document once per unique uploaded file
- Ask questions about the document
- View the retrieved context chunks used for the answer
- Reuse the indexed document on later reruns instead of reprocessing it unnecessarily

## Current system flow

```text
PDF upload
   ↓
PDF text extraction
   ↓
Chunking
   ↓
Embedding generation
   ↓
Chroma vector store
   ↓
User question
   ↓
Semantic retrieval
   ↓
Prompt + retrieved context
   ↓
Gemini response
```

## Main components

- Streamlit frontend: [app.py](app.py)
- RAG pipeline: [rag.py](rag.py)
- Benchmark runner: [benchmark.py](benchmark.py)

## Experimental setup

The project compares multiple retrieval configurations across datasets.

### Retriever types
- `mmr`
- `similarity`

### Values tested
- `k`: 5, 10, 20
- `chunk_size`: 256, 512, 1024
- `embedding_model`:
  - `all-MiniLM-L6-v2`
  - `BAAI/bge-small-en-v1.5`
  - `BAAI/bge-base-en-v1.5`

## Evaluation metrics

The benchmark tracks:

- Retrieval correctness
- Answer correctness
- Response time
- Retrieved chunk count
- Indexing time
- Index source (`built` or `cache`)

## Datasets used

The benchmark currently includes three datasets:

### 1. Hall allotment
A highly structured dataset with loads of different information.

- PDF: `data/1e603f08c8d972ef_2nd yr BOYS_HOSTEL_UG2025.pdf`
- Question file: `data/questions/hall_allotment.json`

Sample questions:
- Which hall is assigned to roll number 25CS10132?
- Which hall has the highest number of students?
- Where should Suhaan Aneja report?

### 2. Git
A structured book(only first 3 chapters).
- PDF: `data/6103fbea51a6bef6_progit-8-105.pdf`
- Question file: `data/questions/git.json`

Sample questions:
- Is Git a distributed version control system?
- Which command initializes a new Git repository?
- Which command clones an existing repository?

### 3. Academic rules
An unstructured dataset

Sample questions:
- Who is the Chairman of the Standing Institute Disciplinary Committee?
- Can a student shift to another room in the hall without permission?
- What is the normal duration of a B.Tech. (Hons.) program?

## How to use the chatbot

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a requirements file yet, install the main packages manually:

```bash
pip install streamlit python-dotenv langchain langchain-community langchain-huggingface langchain-chroma langchain-google-genai langchain-text-splitters pypdf
```

### 2. Set up environment variables

Create a `.env` file in the project root with your Gemini API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run the app

```bash
streamlit run app.py
```

Then:
- upload a PDF
- wait for indexing to finish
- ask questions about the document


 
