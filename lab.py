"""
Lab: Text Embeddings & Cosine Similarity
=========================================
In this lab you will:
  1. Call the OpenAI Embeddings API to embed a list of sentences.
  2. Mean-pool the returned embedding vectors and store them.
  3. Implement cosine similarity from scratch.
  4. Build a function that retrieves the K most similar sentences to a query.

Setup
-----
1. Copy `.env.example` to `.env` and fill in your API key.
2. Install dependencies:  pip install -r requirements.txt
3. Complete every section marked TODO.
"""

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Sample corpus — feel free to swap in your own sentences
# ---------------------------------------------------------------------------
SENTENCES = [
    "The mitochondria is the powerhouse of the cell.",
    "Machine learning models learn patterns from data.",
    "Neural networks are inspired by the human brain.",
    "Python is a popular language for data science.",
    "The quick brown fox jumps over the lazy dog.",
    "Deep learning has revolutionized computer vision.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Gradient descent minimizes the loss function iteratively.",
    "The stock market fluctuates based on investor sentiment.",
    "Transformers use self-attention to process sequences.",
]


# ---------------------------------------------------------------------------
# Part 1 — Embed a list of texts using the OpenAI API
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_embeddings(texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    """
    Call the OpenAI Embeddings API and return a list of embedding vectors,
    one per input text.

    Args:
        texts:  List of strings to embed.
        model:  Embedding model to use.

    Returns:
        List of embedding vectors (each vector is a list of floats).

    Hint: client.embeddings.create(...) returns a response whose `.data`
    attribute is a list of objects, each with an `.embedding` field.
    """
    # TODO: Call the API and return the embeddings.
    
    out = []
    for i in texts:
        response = client.embeddings.create(input=i, model=model)

        out.append(response.data[0].embedding)
    return out



# ---------------------------------------------------------------------------
# Part 2 — Mean pooling
# ---------------------------------------------------------------------------
def mean_pool(embeddings: list[list[float]]) -> np.ndarray:
    """
    Given a list of embedding vectors, return a single vector that is the
    element-wise mean across all vectors.

    Args:
        embeddings: List of embedding vectors (all the same length).

    Returns:
        1-D numpy array of shape (embedding_dim,).

    Note: For a single embedding this just returns that embedding unchanged.
    When you have multiple embeddings for a single "document" (e.g. chunked
    text), mean-pooling collapses them into one representation.
    """
    # TODO: Compute and return the mean-pooled vector.
    return np.mean(np.array(embeddings), axis=0)



# ---------------------------------------------------------------------------
# Part 3 — Cosine similarity
# ---------------------------------------------------------------------------
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

        cosine_similarity(a, b) = (a · b) / (||a|| * ||b||)

    Args:
        vec_a: 1-D numpy array.
        vec_b: 1-D numpy array of the same length as vec_a.

    Returns:
        A float in [-1, 1].

    Important: Do NOT use numpy's or scipy's built-in cosine functions.
    Implement the formula above using only basic numpy operations
    (dot product, norm, etc.).
    """
    # TODO: Implement cosine similarity from scratch.
    x = np.dot(vec_a, vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b))
    return x


# ---------------------------------------------------------------------------
# Part 4 — Top-K retrieval
# ---------------------------------------------------------------------------
def top_k_similar(
    query_vec: np.ndarray,
    corpus_vecs: list[np.ndarray],
    corpus_texts: list[str],
    k: int = 3,
) -> list[tuple[str, float]]:
    """
    Given a query vector and a list of corpus vectors, return the K corpus
    entries most similar to the query.

    Args:
        query_vec:    Embedding of the query text (1-D numpy array).
        corpus_vecs:  List of embeddings for each sentence in the corpus.
        corpus_texts: The original sentences corresponding to corpus_vecs.
        k:            Number of results to return.

    Returns:
        List of (sentence, similarity_score) tuples, sorted from most to
        least similar, length == k.

    Hint: Use your cosine_similarity function from Part 3.
    """
    # TODO: Compute similarities and return the top-k results.
    out = []

    for vec, text in zip(corpus_vecs, corpus_texts):
        x = cosine_similarity(vec, query_vec)
        out.append((text, x))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:k]



# ---------------------------------------------------------------------------
# Main — runs the full pipeline; do not modify
# ---------------------------------------------------------------------------
def main():
    print("=== Embedding corpus ===")
    raw_embeddings = get_embeddings(SENTENCES)
    print(f"  Received {len(raw_embeddings)} embeddings, "
          f"each of dimension {len(raw_embeddings[0])}.")

    # For this lab each sentence is its own "document", so mean-pooling over
    # a single vector is a no-op — but the function you wrote should handle
    # the general case of multiple vectors per document.
    corpus_vecs = [mean_pool([emb]) for emb in raw_embeddings]
    print(f"  Built corpus of {len(corpus_vecs)} vectors.\n")

    query = "How do artificial neural networks work?"
    print(f"=== Query: '{query}' ===")
    query_emb = get_embeddings([query])[0]
    query_vec = mean_pool([query_emb])

    results = top_k_similar(query_vec, corpus_vecs, SENTENCES, k=3)
    print("  Top-3 most similar sentences:")
    for rank, (sentence, score) in enumerate(results, 1):
        print(f"  {rank}. [{score:.4f}] {sentence}")


if __name__ == "__main__":
    main()
