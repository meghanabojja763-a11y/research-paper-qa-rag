from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Embedding model used in your RAG system
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def calculate_similarity(generated_answer, reference_answer):
    """
    Calculate semantic similarity between generated answer
    and the expected/reference answer.
    """

    generated_embedding = embedding_model.encode(
        [generated_answer],
        normalize_embeddings=True
    )

    reference_embedding = embedding_model.encode(
        [reference_answer],
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        generated_embedding,
        reference_embedding
    )[0][0]

    return float(similarity)


def retrieval_hit(retrieved_documents, expected_keywords):
    """
    Check whether expected keywords are present
    in the retrieved document chunks.
    """

    retrieved_text = " ".join(
        doc.page_content.lower()
        for doc in retrieved_documents
    )

    found_keywords = 0

    for keyword in expected_keywords:
        if keyword.lower() in retrieved_text:
            found_keywords += 1

    if len(expected_keywords) == 0:
        return 0.0

    return found_keywords / len(expected_keywords)


def evaluate_question(
    question,
    generated_answer,
    reference_answer,
    retrieved_documents,
    expected_keywords,
    similarity_threshold=0.65
):
    """
    Evaluate one question.
    """

    similarity = calculate_similarity(
        generated_answer,
        reference_answer
    )

    retrieval_score = retrieval_hit(
        retrieved_documents,
        expected_keywords
    )

    answer_correct = similarity >= similarity_threshold

    return {
        "question": question,
        "similarity": similarity,
        "retrieval_score": retrieval_score,
        "answer_correct": answer_correct
    }


def evaluate_dataset(
    vectorstore,
    generate_answer_function,
    dataset,
    k=2,
    similarity_threshold=0.65
):
    """
    Evaluate the complete question-answering system.
    """

    results = []

    for item in dataset:

        question = item["question"]
        reference_answer = item["reference_answer"]
        expected_keywords = item["expected_keywords"]

        # Retrieve relevant chunks
        retrieved_documents = vectorstore.similarity_search(
            question,
            k=k
        )

        # Generate answer
        generated_answer = generate_answer_function(
            question,
            retrieved_documents
        )

        # Evaluate
        result = evaluate_question(
            question=question,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
            retrieved_documents=retrieved_documents,
            expected_keywords=expected_keywords,
            similarity_threshold=similarity_threshold
        )

        result["generated_answer"] = generated_answer
        result["reference_answer"] = reference_answer

        results.append(result)

    # Calculate overall metrics
    total = len(results)

    if total == 0:
        return {
            "results": [],
            "answer_accuracy": 0,
            "retrieval_hit_rate": 0,
            "average_similarity": 0
        }

    answer_accuracy = (
        sum(result["answer_correct"] for result in results)
        / total
    ) * 100

    retrieval_hit_rate = (
        sum(result["retrieval_score"] >= 0.5 for result in results)
        / total
    ) * 100

    average_similarity = (
        sum(result["similarity"] for result in results)
        / total
    )

    return {
        "results": results,
        "answer_accuracy": answer_accuracy,
        "retrieval_hit_rate": retrieval_hit_rate,
        "average_similarity": average_similarity
    }
