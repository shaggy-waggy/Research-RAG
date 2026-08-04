import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:

    def __init__(
        self,
        chunk_size=512,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2",
        retriever_type="mmr",
        k=15,
        fetch_k=10
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.retriever_type = retriever_type
        self.k = k
        self.fetch_k = fetch_k
        self.db_dir = "./chroma_db"


    def load_and_split_pdf(self, file_path):
        print("Loading PDF...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            raise Exception("No text extracted from PDF")

        print(
            f"Loaded {len(documents)} pages"
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = splitter.split_documents(
            documents
        )

        print(
            f"Created {len(chunks)} chunks"
        )

        return chunks



    def _calculate_file_hash(self, file_path):
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as file:
            while True:
                chunk = file.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()


    def _get_collection_name(self, file_path):
        file_hash = self._calculate_file_hash(file_path)
        config_signature = (
            f"{file_hash}:{self.embedding_model}:"
            f"{self.chunk_size}:{self.chunk_overlap}"
        )
        config_hash = hashlib.sha256(
            config_signature.encode("utf-8")
        ).hexdigest()[:24]
        return f"rag_{config_hash}"


    def create_vector_store(self, file_path):

        print("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model
        )

        collection_name = self._get_collection_name(
            file_path
        )
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=self.db_dir,
            embedding_function=embeddings
        )

        existing_docs = vectorstore.get(limit=1)
        if existing_docs.get("ids"):
            print("Using existing vector database")
            return vectorstore

        chunks = self.load_and_split_pdf(file_path)
        vectorstore.add_documents(chunks)
        print("Vector database created")

        return vectorstore



    def create_chain(self, vectorstore):
        print("Creating RAG chain...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a document question answering assistant.

                    Answer only using the provided context.

                    If the answer is not present,
                    say you do not know.

                    Context:
                    {context}
                    """
                ),
                (
                    "human",
                    "{input}"
                )
            ]
        )
        
        if self.retriever_type is "mmr":
            retriever = vectorstore.as_retriever(
                search_type=self.retriever_type,
                search_kwargs={
                    "k": self.k,
                    "fetch_k": self.fetch_k
                }
            )
        elif self.retriever_type is "similarity":
            retriever = vectorstore.as_retriever(
                search_type=self.retriever_type,
                search_kwargs={
                    "k": self.k
                }
            )
        else:
            raise ValueError(
                f"Unknown retriever type: {self.retriever_type}"
            )

        document_chain = create_stuff_documents_chain(
            llm,
            prompt
        )


        rag_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        return rag_chain



    def process_pdf(self, file_path):
        vectorstore = self.create_vector_store(
            file_path
        )


        chain = self.create_chain(
            vectorstore
        )


        return chain