# rag_pipeline.py

import os

from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from groq import Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# EMBEDDING MODEL
# ============================================================

def get_embedding_model():
    """
    Loads the Sentence Transformer embedding model.

    The embedding model converts text into numerical vectors
    that can be used for semantic similarity search.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


# ============================================================
# CREATE FAISS VECTOR DATABASE
# ============================================================

def create_vectorstore(chunks):
    """
    Creates a FAISS vector database from document chunks.

    Each chunk is converted into an embedding and stored
    in the vector database.
    """

    if not chunks:
        raise ValueError(
            "No document chunks were provided."
        )

    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(
    vectorstore,
    question,
    k=4
):
    """
    Performs semantic similarity search.

    Args:
        vectorstore: FAISS vector database
        question: User's question
        k: Number of relevant chunks to retrieve

    Returns:
        List of relevant document chunks
    """

    if vectorstore is None:
        raise ValueError(
            "Vector database has not been created."
        )

    if not question.strip():
        return []

    documents = vectorstore.similarity_search(
        question,
        k=k
    )

    return documents


# ============================================================
# FORMAT RETRIEVED CONTEXT
# ============================================================

def format_context(documents):
    """
    Converts retrieved document chunks into a formatted
    context string for the LLM.
    """

    if not documents:
        return "No relevant information was retrieved."

    context_parts = []

    for index, document in enumerate(documents):

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "Unknown"
        )

        source_text = document.page_content.strip()

        context_parts.append(
            f"""
SOURCE {index + 1}
Page: {page}
Chunk ID: {chunk_id}

{source_text}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# GET GROQ CLIENT
# ============================================================

def get_groq_client():
    """
    Creates and returns the Groq API client.

    The API key is loaded securely from the .env file.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY was not found.\n\n"
            "Please create a .env file in the project folder "
            "and add:\n\n"
            "GROQ_API_KEY=your_api_key_here"
        )

    return Groq(
        api_key=api_key
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    documents
):
    """
    Generates a context-grounded answer using Groq.

    The LLM is instructed to use only the retrieved
    research paper context.
    """

    if not question.strip():

        return (
            "Please enter a question about "
            "the research paper."
        )

    if not documents:

        return (
            "The answer could not be found in "
            "the uploaded research paper."
        )

    # ---------------------------------------------
    # Get Groq client
    # ---------------------------------------------

    client = get_groq_client()

    # ---------------------------------------------
    # Prepare retrieved context
    # ---------------------------------------------

    context = format_context(documents)

    # ---------------------------------------------
    # RAG Prompt
    # ---------------------------------------------

    prompt = f"""
You are a Research Paper Question Answering Assistant.

Your job is to answer questions about a research paper.

You MUST follow these rules:

1. Use ONLY the research paper context provided below.

2. Do NOT use your own external knowledge.

3. Do NOT make assumptions.

4. Do NOT invent facts, datasets, results, authors,
   methodologies, or conclusions.

5. If the answer is not available in the provided
   context, respond exactly with:

"The answer could not be found in the uploaded research paper."

6. Give a clear and understandable answer.

7. Preserve important technical terminology from
   the research paper.

8. When possible, mention the relevant page number
   in your answer.

9. If information comes from multiple sections,
   combine the information into one coherent answer.

10. Do not mention that you are an AI unless necessary.

--------------------------------------------------
RESEARCH PAPER CONTEXT
--------------------------------------------------

{context}

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
ANSWER
--------------------------------------------------
"""

    # ---------------------------------------------
    # Call Groq
    # ---------------------------------------------

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research paper QA assistant. "
                    "Answer only from the supplied document context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1,

        max_tokens=1200
    )

    # ---------------------------------------------
    # Extract answer
    # ---------------------------------------------

    answer = response.choices[0].message.content

    return answer.strip()


# ============================================================
# GET SOURCE INFORMATION
# ============================================================

def get_sources(documents):
    """
    Extracts source page and chunk information from
    retrieved documents.

    Returns:
        List of dictionaries containing page and chunk IDs.
    """

    sources = []

    if not documents:
        return sources

    seen = set()

    for document in documents:

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "Unknown"
        )

        key = (
            page,
            chunk_id
        )

        if key not in seen:

            seen.add(key)

            sources.append(
                {
                    "page": page,
                    "chunk_id": chunk_id
                }
            )

    return sources