# app.py

import streamlit as st

from pdf_processor import (
    extract_pages_from_pdf,
    create_chunks
)

from rag_pipeline import (
    create_vectorstore,
    retrieve_documents,
    generate_answer,
    get_sources
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="Research Paper QA",

    page_icon="📚",

    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "📚 Research Paper Question Answering System"
)

st.write(
    """
    Upload a research paper and ask questions about
    its objective, methodology, datasets, findings,
    limitations, conclusions, and other contents.
    """
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = None


if "chunks" not in st.session_state:

    st.session_state.chunks = []


if "documents" not in st.session_state:

    st.session_state.documents = []


if "file_name" not in st.session_state:

    st.session_state.file_name = ""


if "messages" not in st.session_state:

    st.session_state.messages = []


if "processed_settings" not in st.session_state:

    st.session_state.processed_settings = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    # --------------------------------------------------------
    # Chunk size
    # --------------------------------------------------------

    chunk_size = st.slider(

        "Chunk Size",

        min_value=500,

        max_value=2000,

        value=1000,

        step=100
    )

    # --------------------------------------------------------
    # Chunk overlap
    # --------------------------------------------------------

    chunk_overlap = st.slider(

        "Chunk Overlap",

        min_value=0,

        max_value=500,

        value=200,

        step=50
    )

    # --------------------------------------------------------
    # Retrieval depth
    # --------------------------------------------------------

    retrieval_k = st.slider(

        "Retrieval Depth (Top-K)",

        min_value=1,

        max_value=10,

        value=4,

        step=1
    )

    st.divider()

    st.subheader("🔧 Technology Stack")

    st.markdown(
        """
        **PDF Extraction**  
        PyMuPDF

        **Chunking**  
        Recursive Character Splitter

        **Embeddings**  
        all-MiniLM-L6-v2

        **Vector Database**  
        FAISS

        **LLM**  
        GPT-OSS 120B

        **API**  
        Groq

        **Interface**  
        Streamlit
        """
    )

    st.divider()

    st.info(
        """
        **Current Configuration**

        Chunk Size: {}
        
        Chunk Overlap: {}
        
        Retrieval Depth: {}
        """.format(
            chunk_size,
            chunk_overlap,
            retrieval_k
        )
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader(
    "📄 Upload Research Paper"
)

uploaded_file = st.file_uploader(

    "Choose a research paper in PDF format",

    type=["pdf"]
)


# ============================================================
# PROCESS PDF BUTTON
# ============================================================

if uploaded_file is not None:

    st.write(
        f"**Selected file:** `{uploaded_file.name}`"
    )

    process_button = st.button(

        "⚡ Process Research Paper",

        type="primary",

        use_container_width=True
    )

    if process_button:

        # ----------------------------------------------------
        # Validate chunk settings
        # ----------------------------------------------------

        if chunk_overlap >= chunk_size:

            st.error(
                "Chunk overlap must be smaller "
                "than chunk size."
            )

            st.stop()

        # ----------------------------------------------------
        # Process PDF
        # ----------------------------------------------------

        with st.spinner(
            "Processing research paper..."
        ):

            try:

                # ============================================
                # STEP 1: Extract PDF text
                # ============================================

                documents = extract_pages_from_pdf(
                    uploaded_file
                )

                if not documents:

                    st.error(
                        """
                        No readable text was found in this PDF.

                        This may be a scanned/image-only PDF.
                        OCR can be added later.
                        """
                    )

                    st.stop()

                # ============================================
                # STEP 2: Create chunks
                # ============================================

                chunks = create_chunks(

                    documents,

                    chunk_size=chunk_size,

                    chunk_overlap=chunk_overlap
                )

                if not chunks:

                    st.error(
                        "No text chunks were created."
                    )

                    st.stop()

                # ============================================
                # STEP 3: Create FAISS vector database
                # ============================================

                vectorstore = create_vectorstore(
                    chunks
                )

                # ============================================
                # STEP 4: Store everything
                # ============================================

                st.session_state.documents = (
                    documents
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.vectorstore = (
                    vectorstore
                )

                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.session_state.messages = []

                st.session_state.processed_settings = (
                    chunk_size,
                    chunk_overlap
                )

                st.success(
                    "✅ Research paper processed successfully!"
                )

            except Exception as e:

                st.error(
                    f"Error while processing PDF:\n\n{e}"
                )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader(
        "📊 Research Paper Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    # --------------------------------------------------------
    # Pages
    # --------------------------------------------------------

    with col1:

        st.metric(
            "Pages",
            len(
                st.session_state.documents
            )
        )

    # --------------------------------------------------------
    # Chunks
    # --------------------------------------------------------

    with col2:

        st.metric(
            "Chunks",
            len(
                st.session_state.chunks
            )
        )

    # --------------------------------------------------------
    # Chunk size
    # --------------------------------------------------------

    with col3:

        st.metric(
            "Chunk Size",
            chunk_size
        )

    # --------------------------------------------------------
    # Retrieval depth
    # --------------------------------------------------------

    with col4:

        st.metric(
            "Top-K",
            retrieval_k
        )

    st.caption(
        f"📄 Paper: {st.session_state.file_name}"
    )


# ============================================================
# QUICK QUESTIONS
# ============================================================

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader(
        "🔍 Quick Questions"
    )

    quick_question = None

    # --------------------------------------------------------
    # Row 1
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "🎯 Objective",
            use_container_width=True
        ):

            quick_question = (
                "What is the main objective "
                "of the research paper?"
            )

    with col2:

        if st.button(
            "🧪 Methodology",
            use_container_width=True
        ):

            quick_question = (
                "What methodology was used "
                "in the research paper?"
            )

    with col3:

        if st.button(
            "📊 Datasets",
            use_container_width=True
        ):

            quick_question = (
                "What datasets were used "
                "in the research paper?"
            )

    # --------------------------------------------------------
    # Row 2
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "📈 Major Findings",
            use_container_width=True
        ):

            quick_question = (
                "What are the major findings "
                "of the research paper?"
            )

    with col2:

        if st.button(
            "⚠️ Limitations",
            use_container_width=True
        ):

            quick_question = (
                "What are the limitations "
                "of the research paper?"
            )

    with col3:

        if st.button(
            "📝 Conclusion",
            use_container_width=True
        ):

            quick_question = (
                "What is the conclusion "
                "of the research paper?"
            )


# ============================================================
# QUESTION INPUT
# ============================================================

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader(
        "💬 Ask a Question"
    )

    user_question = st.text_input(

        "Enter your question:",

        placeholder=(
            "Example: What methodology "
            "was used in this paper?"
        )
    )

    ask_button = st.button(

        "🔎 Ask Question",

        type="primary"
    )

    # --------------------------------------------------------
    # Quick question override
    # --------------------------------------------------------

    if quick_question:

        user_question = quick_question

        ask_button = True


# ============================================================
# QUESTION ANSWERING
# ============================================================

if (

    st.session_state.vectorstore is not None

    and ask_button

    and user_question

):

    try:

        # ====================================================
        # STEP 1: RETRIEVAL
        # ====================================================

        with st.spinner(
            "🔎 Searching the research paper..."
        ):

            retrieved_documents = (
                retrieve_documents(

                    st.session_state.vectorstore,

                    user_question,

                    k=retrieval_k
                )
            )

        if not retrieved_documents:

            st.warning(
                "No relevant information was found "
                "in the research paper."
            )

            st.stop()

        # ====================================================
        # STEP 2: LLM GENERATION
        # ====================================================

        with st.spinner(
            "🤖 Generating grounded answer..."
        ):

            answer = generate_answer(

                user_question,

                retrieved_documents
            )

        # ====================================================
        # STEP 3: SAVE RESULT
        # ====================================================

        st.session_state.messages.append(

            {
                "question": user_question,

                "answer": answer,

                "documents": retrieved_documents
            }
        )

    except Exception as e:

        st.error(
            f"Error while answering the question:\n\n{e}"
        )


# ============================================================
# DISPLAY ANSWERS
# ============================================================

if st.session_state.messages:

    st.divider()

    st.subheader(
        "💡 Answers"
    )

    # Show latest answer first

    for message in reversed(
        st.session_state.messages
    ):

        # ----------------------------------------------------
        # Question
        # ----------------------------------------------------

        st.markdown(
            f"### ❓ {message['question']}"
        )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.markdown(
            message["answer"]
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        sources = get_sources(
            message["documents"]
        )

        st.markdown(
            "#### 📚 Sources"
        )

        displayed_pages = set()

        for source in sources:

            page = source["page"]

            chunk_id = source["chunk_id"]

            source_key = (
                page,
                chunk_id
            )

            if source_key not in displayed_pages:

                displayed_pages.add(
                    source_key
                )

                st.markdown(
                    f"- 📄 **Page {page}** "
                    f"— Chunk {chunk_id}"
                )

        st.divider()


# ============================================================
# VIEW RETRIEVED CONTEXT
# ============================================================

if st.session_state.messages:

    with st.expander(
        "🔎 View Retrieved Context"
    ):

        latest_message = (
            st.session_state.messages[-1]
        )

        for index, document in enumerate(

            latest_message["documents"]
        ):

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            chunk_id = document.metadata.get(
                "chunk_id",
                "Unknown"
            )

            st.markdown(
                f"""
                ### Source {index + 1}

                **Page:** {page}

                **Chunk:** {chunk_id}
                """
            )

            st.write(
                document.page_content
            )

            st.divider()


# ============================================================
# PROJECT EXPLANATION
# ============================================================

with st.expander(
    "ℹ️ How this RAG system works"
):

    st.markdown(
        """
        ### RAG Pipeline

        **1. PDF Ingestion**

        The user uploads a research paper in PDF format.

        **2. Text Extraction**

        PyMuPDF extracts text from each PDF page.

        **3. Chunking**

        The extracted text is divided into smaller
        overlapping chunks.

        **4. Embeddings**

        Sentence Transformer converts each chunk
        into a numerical vector.

        **5. Vector Database**

        FAISS stores the embeddings and performs
        semantic similarity search.

        **6. Retrieval**

        When the user asks a question, the system
        retrieves the most relevant chunks.

        **7. Generation**

        The retrieved context and question are
        provided to GPT-OSS 120B through Groq.

        **8. Grounded Answer**

        The LLM generates an answer using only
        the retrieved research paper context.

        **9. Source Citation**

        The application displays the PDF page
        numbers and chunk IDs used for retrieval.
        """
    )