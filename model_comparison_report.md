# Ollama Cloud Model Comparison Report

- Generated: 2026-03-05 22:05:33
- Models: gpt-oss:20b, gpt-oss:120b, qwen3-coder:480b
- Tasks: 2

## Overall Summary

| Model | Success | Avg Latency (s) | Avg Score |
|---|---:|---:|---:|
| gpt-oss:20b | 2/2 | 3.32 | 70.0 |
| gpt-oss:120b | 2/2 | 5.45 | 77.5 |
| qwen3-coder:480b | 2/2 | 18.62 | 77.5 |

## Detailed Outputs

### RAG explanation for beginners

**Prompt**

```text
Explain Retrieval-Augmented Generation (RAG) for beginners.
Use this exact structure:
1) Definition (1 sentence)
2) Pipeline (3 bullet points)
3) When to use (2 bullet points)
4) One limitation (1 bullet point)
Keep it practical and concise.
```

#### gpt-oss:20b

- Latency: 2.05s
- Score: 55
- Verdict: Weak

```text
1) **Definition**  
   Retrieval‑Augmented Generation (RAG) is a framework that blends a language model with a search engine, letting the model pull relevant information from external documents before generating its response.

2) **Pipeline**  
   - **Retrieve**: Query a document index and fetch the most pertinent passages.  
   - **Encode**: Convert the retrieved passages along with the user question into embeddings processed by the language model.  
   - **Generate**: Produce an answer that stitches together the retrieved facts with the model’s own language skills.

3) **When to use**  
   - To answer questions that require up‑to‑date or domain‑specific facts beyond the model’s training data.  
   - When you want the answer to be traceable to source material for transparency or auditing.

4) **One limitation**  
   - The system can still hallucinate if it retrieves irrelevant or incomplete passages, leading the generator to produce inaccurate or misleading information.
```

#### gpt-oss:120b

- Latency: 2.62s
- Score: 70
- Verdict: Usable

```text
**1) Definition**  
Retrieval‑Augmented Generation (RAG) combines a large language model with a searchable external knowledge base, letting the model fetch relevant documents and then generate answers grounded in that retrieved information.  

**2) Pipeline**  
- **Query Encoding:** The user's prompt is transformed into a dense vector using an encoder.  
- **Document Retrieval:** The vector is used to fetch the top‑k most relevant passages from a datastore (e.g., Wikipedia, company docs).  
- **Fusion & Generation:** The retrieved passages are fed into the language model, which conditions its output on both the original query and the retrieved text.  

**3) When to use**  
- When you need up‑to‑date or domain‑specific facts that the base model may not know.  
- When factual correctness and traceability (citing sources) are more important than pure fluency.  

**4) One limitation**  
- Retrieval quality heavily depends on the coverage and indexing of the external knowledge source; gaps or noisy indexes can lead to incorrect or hallucinated answers.
```

#### qwen3-coder:480b

- Latency: 19.36s
- Score: 70
- Verdict: Usable

```text
1) **Definition**: Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation to produce more accurate and factual responses by first fetching relevant documents from a knowledge source.

2) **Pipeline**:
• Retrieve relevant documents or passages from a database/vector store based on the input query
• Combine the retrieved information with the original prompt to create an enhanced context
• Generate a response using an AI model that considers both the query and retrieved information

3) **When to use**:
• When you need factually accurate responses based on specific, up-to-date information
• When working with domain-specific content that requires precise knowledge retrieval

4) **One limitation**: RAG can be slow and computationally expensive due to the additional retrieval step before generation.
```

### Mini Python RAG skeleton

**Prompt**

```text
Write a minimal Python skeleton for a RAG pipeline using placeholders.
Requirements:
- Use function names: embed_query, retrieve_top_k, build_prompt, ask_model
- Show data flow from query to final answer
- Include comments for where vector DB and LLM calls happen
- Keep it to 12-20 lines of code
```

#### gpt-oss:20b

- Latency: 4.59s
- Score: 85
- Verdict: Strong

```text
```python
# Minimal RAG skeleton
def embed_query(query):
    # Convert the query to an embedding (placeholder for the embedding model)
    return "embedding_of_" + query

def retrieve_top_k(embedding, k=3):
    # Query the vector DB for top‑k similar documents (placeholder)
    return [f"doc{index}" for index in range(1, k+1)]

def build_prompt(docs, query):
    # Build the prompt from retrieved docs and the user query (placeholder)
    context = "\n".join(docs)
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

def ask_model(prompt):
    # Send the prompt to an LLM and return the response (placeholder)
    return "Answer from LLM based on prompt"

def rag_pipeline(query):
    embedding = embed_query(query)
    docs      = retrieve_top_k(embedding)
    prompt    = build_prompt(docs, query)
    answer    = ask_model(prompt)
    return answer
```
```

#### gpt-oss:120b

- Latency: 8.28s
- Score: 85
- Verdict: Strong

```text
```python
import typing                                   # (optional) type hints

def embed_query(query: str) -> list:           # → compute/query embedding
    return [0.0] * 768                         # placeholder vector

def retrieve_top_k(vec: list, k: int):         # ← vector‑DB similarity search
    return ["doc 1", "doc 2", "doc 3"][:k]     # mock retrieved chunks

def build_prompt(query: str, docs: list) -> str:
    return query + "\n---\n" + "\n".join(docs)

def ask_model(prompt: str) -> str:             # → LLM generation call
    return "🡒 generated answer"                  # placeholder response

def rag_pipeline(query: str, k: int = 5) -> str:
    docs   = retrieve_top_k(embed_query(query), k)   # query → docs
    answer = ask_model(build_prompt(query, docs))   # docs + query → answer
    return answer

if __name__ == "__main__":
    print(rag_pipeline("What is retrieval‑augmented generation?"))
```
```

#### qwen3-coder:480b

- Latency: 17.88s
- Score: 85
- Verdict: Strong

```text
```python
def embed_query(query):
    # Placeholder for embedding generation
    # In practice: call embedding model (e.g., SentenceTransformer, OpenAI Embeddings)
    return f"embedding_of_{query}"

def retrieve_top_k(embedding, k=3):
    # Placeholder for vector database retrieval
    # In practice: query vector DB (e.g., Pinecone, Weaviate, FAISS)
    return ["doc1", "doc2", "doc3"]  # retrieved documents

def build_prompt(query, documents):
    # Combine query with retrieved context
    context = "\n".join(documents)
    return f"Context: {context}\n\nQuestion: {query}"

def ask_model(prompt):
    # Placeholder for LLM call
    # In practice: call LLM API (e.g., OpenAI, HuggingFace, local model)
    return "This is a generated answer based on the context."

# RAG Pipeline Execution
query = "What is the capital of France?"
embedding = embed_query(query)                    # Generate query embedding
docs = retrieve_top_k(embedding, k=3)            # Retrieve relevant documents
prompt = build_prompt(query, docs)               # Build contextual prompt
answer = ask_model(prompt)                       # Generate final answer
print(answer)
```
```
