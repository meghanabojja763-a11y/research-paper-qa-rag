# pdf_processor.py

import fitz

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_pages_from_pdf(pdf_file):
    """
    Extract text from every page of a PDF.

    Each page is stored as a LangChain Document.
    Page number is preserved in metadata for citation.
    """

    pdf_bytes = pdf_file.getvalue()

    pdf_document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    documents = []

    for page_number, page in enumerate(pdf_document):

        text = page.get_text("text")

        text = text.strip()

        if text:

            document = Document(
                page_content=text,
                metadata={
                    "page": page_number + 1
                }
            )

            documents.append(document)

    pdf_document.close()

    return documents


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================

def create_chunks(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Split extracted page text into smaller overlapping chunks.

    Default:
        chunk_size = 1000
        chunk_overlap = 200
    """

    if not documents:
        return []

    if chunk_overlap >= chunk_size:

        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    # Add chunk IDs
    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index + 1

    return chunks