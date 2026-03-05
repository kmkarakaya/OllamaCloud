# Ollama Cloud Model Comparison Report

- Generated: 2026-03-05 23:27:57
- Models: gpt-oss:20b, gpt-oss:120b, qwen3-coder:480b
- Tasks: 2

## Overall Summary

| Model | Success | Avg Latency (s) | Avg Score |
|---|---:|---:|---:|
| gpt-oss:20b | 2/2 | 7.17 | 77.5 |
| gpt-oss:120b | 2/2 | 5.67 | 77.5 |
| qwen3-coder:480b | 2/2 | 5.15 | 85.0 |

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

- Latency: 5.15s
- Score: 70
- Verdict: Usable

```text
**1) Definition**  
Retrieval‑Augmented Generation (RAG) is a technique that first fetches relevant documents from an external database and then uses a language model to generate an answer based on both the query and the retrieved content.

**2) Pipeline**  
- Encode the user’s question and retrieve the top‑k most relevant text passages from a pre‑indexed knowledge base.  
- Concatenate the question with the retrieved passages and feed them into a transformer‑based generation model.  
- Produce the final answer, optionally tagging or highlighting the source passages for transparency.

**3) When to use**  
- When the answer must incorporate up‑to‑date or domain‑specific facts that are unlikely to be memorized by the language model.  
- When precise, fact‑checked responses are required (e.g., medical, legal, or compliance queries).

**4) One limitation**  
- If the retrieval step retrieves noisy or irrelevant documents, the generator can produce inaccurate or hallucinated information.
```

#### gpt-oss:120b

- Latency: 3.31s
- Score: 70
- Verdict: Usable

```text
1) **Definition (1 sentence)**  
Retrieval‑Augmented Generation (RAG) combines a search over an external knowledge source with a language model that generates text, letting the model ground its output in up‑to‑date or specialized information.

2) **Pipeline (3 bullet points)**  
- **Retrieve:** A query is sent to a vector‑store or traditional search engine, returning the most relevant documents or passages.  
- **Combine:** The retrieved texts are concatenated with the original prompt (often with special tokens) and fed to the generative model.  
- **Generate:** The language model produces a response that incorporates the retrieved evidence, optionally citing sources.

3) **When to use (2 bullet points)**  
- When you need factual accuracy on dynamic or domain‑specific data that the base model wasn’t trained on (e.g., product catalogs, recent news).  
- When you want to keep the model lightweight while still accessing a large external knowledge base.

4) **One limitation (1 bullet point)**  
- The quality of the final answer is heavily dependent on the relevance of the retrieved documents; poor retrieval can lead to incorrect or nonsensical generations.
```

#### qwen3-coder:480b

- Latency: 7.87s
- Score: 85
- Verdict: Strong

```text
1) **Definition**: Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation to produce more accurate and factual responses by pulling relevant data from external sources before generating answers.

2) **Pipeline**:
• **Retrieve**: Search through a knowledge base or database to find relevant documents/information related to the query
• **Augment**: Combine the retrieved information with the original query to create enhanced input context
• **Generate**: Use an AI model to produce a response based on both the query and retrieved information

3) **When to use**:
• When you need factually accurate responses that require up-to-date or specific organizational knowledge
• When building QA systems, chatbots, or content generators that must avoid hallucination and cite reliable sources

4) **One limitation**:
• Retrieval quality directly impacts final output - if the system retrieves irrelevant or poor-quality information, the generated response will be compromised regardless of how good the language model is
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

- Latency: 9.18s
- Score: 85
- Verdict: Strong

```text
```python
def embed_query(query):
    # Vector‑DB: turn the query into an embedding vector
    return []                       # placeholder


def retrieve_top_k(embedding, k=5):
    # Vector‑DB: fetch the k most relevant documents
    return []                       # placeholder


def build_prompt(docs, query):
    # Build the prompt that will be sent to the LLM
    return ""                        # placeholder


def ask_model(prompt):
    # LLM call: generate the final answer
    return ""                        # placeholder


def rag_pipeline(query, k=5):
    # Query → embed → retrieve → build prompt → ask LLM → answer
    return ask_model(build_prompt(
        retrieve_top_k(embed_query(query), k), query
    ))


print(rag_pipeline("What is RAG?"))
```

*The comments mark where the vector‑database and LLM calls would normally occur. The flow from user query to the final answer follows the functions in order.*
```

#### gpt-oss:120b

- Latency: 8.03s
- Score: 85
- Verdict: Strong

```text
```python
# Minimal RAG pipeline skeleton
def embed_query(q: str) -> list:          # → embedding vector
    # TODO: call embedding model (e.g., OpenAI, Sentence‑Transformers)
    return []

def retrieve_top_k(e: list, k: int = 5) -> list:   # → top‑k documents
    # TODO: vector‑DB similarity search (FAISS, Pinecone, etc.)
    return []

def build_prompt(docs: list, q: str) -> str:
    # Assemble retrieved docs and the original query into an LLM prompt
    return ""

def ask_model(p: str) -> str:
    # TODO: send prompt to a LLM (GPT‑4, Claude, etc.) and get the answer
    return ""

def rag(q: str) -> str:
    """End‑to‑end data flow: query → answer."""
    return ask_model(build_prompt(retrieve_top_k(embed_query(q)), q))
```
```

#### qwen3-coder:480b

- Latency: 2.43s
- Score: 85
- Verdict: Strong

```text
```python
def embed_query(query):
    # Placeholder: embed query using embedding model
    return f"embedded_{query}"

def retrieve_top_k(query_embedding, k=3):
    # Placeholder: search vector DB for top-k similar documents
    return [f"doc_{i}" for i in range(k)]

def build_prompt(query, documents):
    # Placeholder: format query and retrieved docs into prompt
    return f"Query: {query}\nDocs: {documents}"

def ask_model(prompt):
    # Placeholder: call LLM API with prompt
    return "Generated answer based on retrieved context"

# RAG Pipeline
query = "What is retrieval augmented generation?"
query_embedding = embed_query(query)                    # Embedding step
docs = retrieve_top_k(query_embedding, k=3)            # Vector DB lookup
prompt = build_prompt(query, docs)                      # Prompt construction
answer = ask_model(prompt)                              # LLM generation
print(answer)
```
```
