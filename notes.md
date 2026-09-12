# RAG (Retrieval-Augmented Generation) Comprehensive Notes (1-13)

---

## 🗺️ Sequential End-to-End Production Project Lifecycle Index
> **How a Proper Production RAG & Agentic Project Flows Sequentially by Topic**
>
> In real-world enterprise engineering, an AI system follows a disciplined sequential pipeline: from collecting and ingesting multi-format data, to token/meaning-aware chunking, vector embeddings, database indexing, pre-retrieval query enhancement, hybrid search & re-ranking, LCEL/pre-built RAG chains, and multimodal expansion.

| Project Stage | Pipeline Phase | Core Techniques & Focus | Direct Section Link |
| :---: | :--- | :--- | :--- |
| **Stage 1** | **Multi-Source Data Ingestion & Parsing** | Ingesting raw unstructured, semi-structured, and tabular data | [1. Data Ingestion](#data-ingestion) • [2. PDF Parsing](#pdf-parsing) • [3. Word Docs](#word-parsing) • [4. CSV/Excel](#csv-excel-parsing) • [5. JSON](#json-parsing) • [6. Relational DB](#database-parsing) |
| **Stage 2** | **Document Splitting & Chunking Strategies** | Boundary preservation, token limits, and meaning-aware splitting | [10 Core Chunking Strategies](#chunking-strategies) • [Semantic Chunking](#semantic-chunking) |
| **Stage 3** | **Vector Embeddings & Representation** | High-dimensional semantic vectors and dense embeddings | [Embedding Models (OpenAI & HF)](#embedding-models) |
| **Stage 4** | **Vector Storage & Database Indexing** | Local vector stores vs cloud databases, CRUD, and distance metrics | [Vector Stores vs Vector Databases](#vector-store-vs-db) • [ChromaDB](#chroma-db) • [FAISS](#faiss) • [Pinecone](#pinecone) • [InMemory](#inmemory-store) • [Qdrant](#qdrant) • [Distance Metrics](#distance-metrics) |
| **Stage 5** | **Pre-Retrieval Query Enhancement** | Bridging semantic gaps, multi-hop sub-queries, and hypothetical answers | [12.1 Query Expansion](#query-expansion) • [12.2 Query Decomposition](#query-decomposition) • [12.3 HyDE (Hypothetical Doc Embeddings)](#hyde) |
| **Stage 6** | **Advanced Retrieval & Precision Ranking** | Dense + Sparse hybrid fusion, cross-encoder re-ranking, and diversity | [11.1 Dense + Sparse Hybrid Search](#dense-sparse-retriever) • [11.2 Re-ranking](#reranking) • [11.3 MMR (Maximal Marginal Relevance)](#mmr) • [11.4 Production Search Strategies](#production-search) |
| **Stage 7** | **RAG Chain Construction & Memory** | LCEL composition, conversational memory, and built-in retrieval helpers | [Custom LCEL RAG Chain](#lcel-rag-chain) • [Conversational RAG (History)](#conversational-rag) • [Modern Classic Retrieval Chain](#classic-rag-chain) • [RAG Chain Types Comparison](#chain-types-comparison) |
| **Stage 7.1** | **Chain Architecture Decision Framework** | 🎯 **Deep Dive: When to Use vs. When NOT to Use `format_docs`** (LCEL vs. Pre-built Helpers Comparison Matrix) | [format_docs Decision Guide](#format-docs-guide) |
| **Stage 8** | **Multimodal RAG & Visual Intelligence** | Cross-modal text-to-image retrieval, CLIP joint space, and Vision LLMs | [13. Multimodal RAG (notes2.md)](file:///c:/Users/DELL/Desktop/rag_praacties/notes2.md#13-multimodal-rag-07_multimodle-rag) |

---

## 📋 Table of Contents (Module-by-Module)

1. [Stage 1: Data Ingestion & Splitting](#data-ingestion)
   * [Stage 2: 10 Core Chunking Strategies in RAG](#chunking-strategies)

2. [Stage 1: PDF Parsing](#pdf-parsing)

3. [Stage 1: Word Document Parsing](#word-parsing)

4. [Stage 1: CSV & Excel Structured Parsing](#csv-excel-parsing)

5. [Stage 1: JSON Parsing](#json-parsing)

6. [Stage 1: Database Parsing](#database-parsing)

7. [Stage 3: Embedding Models](#embedding-models)

8. [Stage 4: Vector Databases](#vector-databases)
   * [ChromaDB (`langchain_chroma.Chroma`)](#chroma-db)
   * [FAISS (`langchain_community.vectorstores.FAISS`)](#faiss)
   * [Pinecone (`langchain_pinecone.PineconeVectorStore`)](#pinecone)
   * [InMemoryVectorStore](#inmemory-store)
   * [Qdrant (`langchain_qdrant.QdrantVectorStore`)](#qdrant)
   * [Vector Distance Metrics & Similarity Scores](#distance-metrics)

9. [Stage 7: RAG Chains & Conversational Memory](#rag-chains)
   * [1. LLM / Model Initialization Methods](#llm-init-methods)
   * [2. Custom RAG Chain using LCEL (LangChain Expression Language)](#lcel-rag-chain)
   * [3. Conversational RAG Chain (With History/Memory)](#conversational-rag)
   * [4. Modern RAG Chain (Using LangChain Classic Retrieval Chain)](#classic-rag-chain)
   * [Stage 7.1: 5. When to Use vs. When NOT to Use `format_docs` in LangChain](#format-docs-guide)

10. [Stage 2: Semantic Chunking](#semantic-chunking)
    * [RAG Chain Types Comparison](#chain-types-comparison)
    * [Vector Store vs Vector Database](#vector-store-vs-db)

11. [Stage 6: Hybrid Search & Re-ranking](#hybrid-search)
    * [11.1 Hybrid Retriever – Dense & Sparse Combination](#dense-sparse-retriever)
    * [11.2 Re-ranking Hybrid Search Strategies](#reranking)
    * [11.3 Maximal Marginal Relevance - MMR](#mmr)
    * [11.4 RAG Search Strategies & Production Search Pipelines](#production-search)

12. [Stage 5: Query Enhancement & Advanced RAG](#query-enhancement)
    * [12.1 Query Expansion](#query-expansion)
    * [12.2 Query Decomposition](#query-decomposition)
    * [12.3 Hypothetical Document Embeddings (HyDE)](#hyde)

13. [Stage 8: Multimodal RAG (Transferred to notes2.md)](file:///c:/Users/DELL/Desktop/rag_praacties/notes2.md#13-multimodal-rag-07_multimodle-rag)

---

## 1. Data Ingestion & Splitting (`1-dataingestion.ipynb`) <a id="data-ingestion" name="data-ingestion"></a>
### Imports
```python
import tempfile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
```

<details>
<summary><b>💡 Helper Pattern: Programmatically Generating Temporary Sample Files</b></summary>

When testing loaders (like `DirectoryLoader`) without cluttering your project folder with hardcoded files, you can create temporary files using Python's `tempfile` module:

```python
import tempfile

# 1. Create temporary directory
temp_dir = tempfile.mkdtemp()

sample_docs = [
    "Artificial Intelligence (AI) is transforming industries.",
    "Machine Learning is a subset of AI focusing on data and algorithms.",
    "Retrieval-Augmented Generation (RAG) enhances LLMs with dynamic external retrieval."
]

# 2. Write sample documents to temporary files
for i, doc in enumerate(sample_docs):
    with open(f"{temp_dir}/doc_{i}.txt", "w") as f:
        f.write(doc)

print(f"Sample documents created in: {temp_dir}")
```
*   **Why use this?** Ideal for unit tests, notebooks, and quick experiments because it creates a isolated sandbox directory that can be safely discarded later.
</details>

### How to Use

#### Single File Loading (`TextLoader`)
```python
# Load single text file and split recursively
loader = TextLoader("data/sample.txt")
documents = loader.load()

# 🔹 KEY CONFIG: chunk_size=500, chunk_overlap=50 (10% overlap ratio rule of thumb)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]  # Preserves paragraph & sentence structures
)
chunks = text_splitter.split_documents(documents)
```

#### Bulk Folder Ingestion (`DirectoryLoader`)
```python
# Bulk load all matching text files from a directory path
dir_loader = DirectoryLoader(
    path=temp_dir,          # Directory path containing documents (e.g., "data/" or temp_dir)
    glob="*.txt",           # File pattern matching
    loader_cls=TextLoader,  # Underlying loader class used for each file
    show_progress=True      # Show progress bar during bulk loading
)
bulk_documents = dir_loader.load()
```

### What They Do
*   `Document`: LangChain's base class storing raw text (`page_content`) and arbitrary key-value dict metadata (`metadata`).
*   `DirectoryLoader`: Scans a directory for matching files (`glob="*.txt"`) and loads them concurrently using a specified single-file loader (`loader_cls=TextLoader`).
*   `RecursiveCharacterTextSplitter`: <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Best default splitter</mark>. Recursively splits text using a hierarchy of separators (`\n\n`, `\n`, `" "`, `""`) to keep paragraphs and sentences visually intact.
*   `CharacterTextSplitter`: Splits text rigidly based on a single character separator (e.g. `\n\n`), risking oversized chunks if separators are far apart.
*   `TokenTextSplitter`: Splits text strictly by token count (using OpenAI `tiktoken`) to guarantee chunk sizes fit exact LLM context boundaries.
*   `TextLoader` / `DirectoryLoader`: Loads a single plain text file / bulk-loads matching document files from a folder directory.

### 💡 Advanced Best Practices & Key Insights:
*   **Chunk Overlap Strategy**: Always set `chunk_overlap` between <mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">10% to 20% of `chunk_size`</mark>. This prevents losing critical semantic context across chunk boundary splits.
*   **Metadata Enrichment**: Always inject custom metadata attributes (e.g., `source`, `creation_date`, `category`) onto each Document during ingestion for precise metadata filtering in vector databases.

<br>

### 🧩 10 Core Chunking Strategies in RAG <a id="chunking-strategies" name="chunking-strategies"></a>

> 💡 **For RAG, the most commonly useful starting points are:**
> **Recursive + overlap**, **semantic**, and **document-structure-based chunking**.

#### Summary Matrix:
| # | Strategy | Core Mechanism | Best For | Trade-offs & Limitations |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Fixed-size chunking** | Split strictly every $N$ characters/tokens | Rapid baseline tests | Slices words & sentences mid-thought |
| 2 | **Sentence-based chunking** | Split at sentence punctuation (`.`, `!`, `?`) | Factoid Q&A, statement search | Uneven chunk lengths; loses paragraph context |
| 3 | **Paragraph-based chunking** | Split at double newlines (`\n\n`) | Articles, blogs, narratives | Paragraphs vary wildly in length |
| 4 | **Recursive chunking** | Hierarchical separators (`\n\n` → `\n` → `" "` → `""`) | General prose (Default choice) | Complex technical documents can still fragment |
| 5 | **Semantic chunking** | Split on sentence embedding distance spikes | Dense technical / academic text | High computation cost ($N$ embedding calls) |
| 6 | **Document-structure chunking** | Split on headers (`#`, `##`), HTML tags, tables | Markdown docs, manuals, code | Irregular chunk sizes; requires structural markup |
| 7 | **Sliding-window chunking** | Fixed window size + fixed stride overlap | High-continuity document streams | Substantial data redundancy in vector index |
| 8 | **Token-based chunking** | Split strictly by tokenizer token limit | LLM context window budgeting | Disregards grammatical sentence boundaries |
| 9 | **Agentic/LLM-based chunking** | Prompt LLM to extract cohesive sections | Unstructured messy data | High API latency and financial cost |
| 10 | **Hybrid chunking** | Structure + Recursive/Semantic + Token limits | Enterprise-grade production RAG | Multi-step pipeline implementation overhead |

---

#### 1. Fixed-size chunking
Splits text every $N$ characters (or words) with an optional overlap, without taking grammatical or linguistic structure into account.
* **Best used for:** Quick baseline tests or uniform flat data where semantic boundaries are unimportant.
* **Risk:** Cuts words and sentences in half, causing context fragmentation and hallucinations.

<details>
<summary><b>Code & Example: Fixed-size Chunking</b></summary>

```python
from langchain_text_splitters import CharacterTextSplitter

text = (
    "LangChain is an orchestration framework for LLMs. It connects models "
    "to external data sources and enables retrieval-augmented generation. "
    "Chroma and FAISS are common vector stores used for fast similarity search."
)

splitter = CharacterTextSplitter(
    separator="",          # Hard character split
    chunk_size=60,         # Exact character count
    chunk_overlap=10       # Fixed overlap
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} [{len(chunk)} chars]: '{chunk}'")
```

**Output Example:**
```text
Chunk 1 [60 chars]: 'LangChain is an orchestration framework for LLMs. It connect'
Chunk 2 [60 chars]: 'connects models to external data sources and enables retriev'
Chunk 3 [60 chars]: 'retrieval-augmented generation. Chroma and FAISS are common '
Chunk 4 [48 chars]: 'common vector stores used for fast similarity search.'
```
</details>

---

#### 2. Sentence-based chunking
Splits text directly along sentence boundaries (using punctuation marks like `.`, `!`, `?` or NLP tokenizers from NLTK/spaCy).
* **Best used for:** Precise fact-checking, statement verification, and sentence-level quote retrieval.
* **Risk:** Individual sentences frequently lack sufficient context (e.g., resolving pronouns like "it", "they", or "this").

<details>
<summary><b>Code & Example: Sentence-based Chunking</b></summary>

```python
import re

text = (
    "Retrieval-Augmented Generation enhances LLM capability! "
    "It fetches relevant knowledge from external vector databases. "
    "Does this prevent hallucinations? Yes, by grounding answers in retrieved source text."
)

# Sentence boundary regex or NLTK/spaCy sentence tokenizer
sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

for i, sent in enumerate(sentences, 1):
    print(f"Chunk {i} (Sentence): {sent}")
```

**Output Example:**
```text
Chunk 1 (Sentence): Retrieval-Augmented Generation enhances LLM capability!
Chunk 2 (Sentence): It fetches relevant knowledge from external vector databases.
Chunk 3 (Sentence): Does this prevent hallucinations?
Chunk 4 (Sentence): Yes, by grounding answers in retrieved source text.
```
</details>

---

#### 3. Paragraph-based chunking
Uses natural paragraph delimiters (usually double newlines `\n\n`) to segment text, maintaining the author's original unit of thought.
* **Best used for:** Well-formatted editorial content, blog posts, essays, and reports where paragraphs represent cohesive ideas.
* **Risk:** Paragraph lengths vary wildly; one paragraph may be 20 tokens while another is 2,000 tokens, exceeding LLM context limits.

<details>
<summary><b>Code & Example: Paragraph-based Chunking</b></summary>

```python
from langchain_text_splitters import CharacterTextSplitter

text = """Artificial Intelligence has revolutionized natural language processing. Modern transformer architectures allow models to understand contextual relationships across vast amounts of text.

Retrieval-Augmented Generation (RAG) is a prominent architecture that combines information retrieval with text generation. By grounding responses in external knowledge, RAG dramatically reduces hallucinations.

Vector databases act as the memory layer in RAG systems, enabling sub-second semantic retrieval across millions of embeddings."""

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,         # Minimum target before splitting
    chunk_overlap=0
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} (Paragraph) ---\n{chunk}\n")
```

**Output Example:**
```text
--- Chunk 1 (Paragraph) ---
Artificial Intelligence has revolutionized natural language processing. Modern transformer architectures allow models to understand contextual relationships across vast amounts of text.

--- Chunk 2 (Paragraph) ---
Retrieval-Augmented Generation (RAG) is a prominent architecture that combines information retrieval with text generation. By grounding responses in external knowledge, RAG dramatically reduces hallucinations.

--- Chunk 3 (Paragraph) ---
Vector databases act as the memory layer in RAG systems, enabling sub-second semantic retrieval across millions of embeddings.
```
</details>

---

#### 4. Recursive chunking
Tries a prioritized hierarchy of separators sequentially (`\n\n` → `\n` → `" "` → `""`). It only moves to finer separators if a chunk exceeds the target size, keeping larger structural units (paragraphs, then sentences) intact whenever possible.
* **Best used for:** General prose, documentation, articles, and default RAG pipelines (recommended baseline).
* **Risk:** Dense technical sections without standard paragraph breaks can still end up fragmented.

<details>
<summary><b>Code & Example: Recursive Chunking</b></summary>

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """LangChain provides modular abstractions. It simplifies building LLM apps.

RAG combines retrieval with generation:
- Dense retrieval uses vector embeddings.
- Sparse retrieval uses keyword algorithms like BM25.
Combining both creates hybrid search."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} ({len(chunk)} chars):\n\"{chunk}\"\n")
```

**Output Example:**
```text
Chunk 1 (74 chars):
"LangChain provides modular abstractions. It simplifies building LLM apps."

Chunk 2 (112 chars):
"RAG combines retrieval with generation:
- Dense retrieval uses vector embeddings.
- Sparse retrieval uses keyword"

Chunk 3 (85 chars):
"- Sparse retrieval uses keyword algorithms like BM25.
Combining both creates hybrid search."
```
</details>

---

#### 5. Semantic chunking
Computes embeddings for consecutive sentences and measures their cosine distance. When the semantic distance between adjacent sentences spikes past a calculated statistical threshold (percentile or standard deviation), it places a chunk boundary.
* **Best used for:** Dense multi-topic documents, research papers, and technical transcripts where topics shift unpredictably.
* **Risk:** Computationally expensive at ingestion time ($N$ sentence embedding inference calls).

<details>
<summary><b>Code & Example: Semantic Chunking</b></summary>

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

text = """LangChain is a framework for building applications with LLMs.
It provides modular abstractions to combine LLMs with vector databases like Chroma and Pinecone.
You can create chains, agents, memory, and retrievers.
The Eiffel Tower is located on the Champ de Mars in Paris, France.
France is one of the most visited tourist destinations in the world."""

# Splits when cosine distance between consecutive sentences exceeds 95th percentile
chunker = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95.0
)

docs = chunker.create_documents([text])
for i, doc in enumerate(docs, 1):
    print(f"=== Semantic Chunk {i} ===\n{doc.page_content}\n")
```

**Output Example:**
```text
=== Semantic Chunk 1 ===
LangChain is a framework for building applications with LLMs. It provides modular abstractions to combine LLMs with vector databases like Chroma and Pinecone. You can create chains, agents, memory, and retrievers.

=== Semantic Chunk 2 ===
The Eiffel Tower is located on the Champ de Mars in Paris, France. France is one of the most visited tourist destinations in the world.
```
</details>

---

#### 6. Document-structure chunking
Leverages the inherent layout and syntax of documents — such as Markdown headers (`#`, `##`, `###`), HTML tags (`<section>`, `<table>`), code ASTs (classes, functions), or JSON keys. It attaches structural breadcrumbs directly into chunk metadata.
* **Best used for:** Technical documentation, API specs, developer docs, GitHub repositories, and structured reports.
* **Risk:** Chunk size is determined entirely by author formatting; long sections without subheaders may still need secondary splitting.

<details>
<summary><b>Code & Example: Document-structure Chunking</b></summary>

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """# Machine Learning
Machine learning algorithms build mathematical models based on sample training data.

## Supervised Learning
Supervised algorithms require input data paired with corresponding ground-truth labels.

### Classification
Predicts discrete category labels like spam detection.

## Unsupervised Learning
Discovers inherent groupings or patterns in unlabeled data."""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)

splits = markdown_splitter.split_text(markdown_text)
for i, doc in enumerate(splits, 1):
    print(f"Chunk {i} | Metadata: {doc.metadata}")
    print(f"Content:\n{doc.page_content}\n")
```

**Output Example:**
```text
Chunk 1 | Metadata: {'Header 1': 'Machine Learning'}
Content:
# Machine Learning
Machine learning algorithms build mathematical models based on sample training data.

Chunk 2 | Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Supervised Learning'}
Content:
## Supervised Learning
Supervised algorithms require input data paired with corresponding ground-truth labels.

Chunk 3 | Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Supervised Learning', 'Header 3': 'Classification'}
Content:
### Classification
Predicts discrete category labels like spam detection.

Chunk 4 | Metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Unsupervised Learning'}
Content:
## Unsupervised Learning
Discovers inherent groupings or patterns in unlabeled data.
```
</details>

---

#### 7. Sliding-window chunking
Generates overlapping chunks by sliding a fixed-size window forward by a smaller step (stride). A window of size $W$ with stride $S$ produces an overlap of $W - S$ tokens across consecutive chunks, ensuring transitions between boundaries are never lost.
* **Best used for:** Continuous text streams, conversation transcripts, medical records, or legal contracts where context shearing is unacceptable.
* **Risk:** High storage and vector compute overhead due to repetitive text redundancy.

<details>
<summary><b>Code & Example: Sliding-window Chunking</b></summary>

```python
def sliding_window_chunking(text: str, window_size: int = 50, stride: int = 30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), stride):
        chunk_words = words[i : i + window_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + window_size >= len(words):
            break
    return chunks

sample = (
    "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda Mu "
    "Nu Xi Omicron Pi Rho Sigma Tau Upsilon Phi Chi Psi Omega"
)

chunks = sliding_window_chunking(sample, window_size=10, stride=6)  # 4 words overlap
for i, chunk in enumerate(chunks, 1):
    print(f"Window {i}: {chunk}")
```

**Output Example:**
```text
Window 1: Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa
Window 2: Eta Theta Iota Kappa Lambda Mu Nu Xi Omicron Pi
Window 3: Nu Xi Omicron Pi Rho Sigma Tau Upsilon Phi Chi
Window 4: Tau Upsilon Phi Chi Psi Omega
```
</details>

---

#### 8. Token-based chunking
Splits text according to explicit token counts using the target LLM's exact BPE tokenizer (e.g., `tiktoken` for OpenAI `cl100k_base` / `o200k_base`).
* **Best used for:** Strict LLM context window budgeting, preventing API token limit errors, and accurate cost tracking.
* **Risk:** May split in the middle of words or sentences if not combined with recursive fallback characters.

<details>
<summary><b>Code & Example: Token-based Chunking</b></summary>

```python
from langchain_text_splitters import TokenTextSplitter

text = (
    "TokenTextSplitter splits documents strictly based on the token count "
    "generated by BPE tokenizers like tiktoken. This is essential when building "
    "RAG prompts with fixed LLM context window constraints."
)

token_splitter = TokenTextSplitter(
    chunk_size=15,      # Split every 15 tokens
    chunk_overlap=3,    # 3 tokens overlap
    encoding_name="cl100k_base"
)

chunks = token_splitter.split_text(text)
for i, chunk in enumerate(chunks, 1):
    print(f"Token Chunk {i}:\n'{chunk}'\n")
```

**Output Example:**
```text
Token Chunk 1:
'TokenTextSplitter splits documents strictly based on the token count'

Token Chunk 2:
' the token count generated by BPE tokenizers like tiktoken. This'

Token Chunk 3:
' tiktoken. This is essential when building RAG prompts with fixed'

Token Chunk 4:
' prompts with fixed LLM context window constraints.'
```
</details>

---

#### 9. Agentic / LLM-based chunking
Employs an LLM as an intelligent chunking agent. The model reads the document, reasons about semantic boundaries, and outputs clean, self-contained sections accompanied by synthesized context, summaries, or standalone chunk titles.
* **Best used for:** Complex, messy, or highly technical documents where rule-based splitters fail (e.g., mixed tables, contracts, research summaries).
* **Risk:** Substantial inference cost and higher processing latency per document during data ingestion.

<details>
<summary><b>Code & Example: Agentic/LLM-based Chunking</b></summary>

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

class ChunkOutput(BaseModel):
    chunk_title: str = Field(description="Descriptive title for this chunk")
    chunk_content: str = Field(description="Self-contained chunk text with context preserved")

class DocumentChunks(BaseModel):
    chunks: List[ChunkOutput]

parser = JsonOutputParser(pydantic_object=DocumentChunks)

prompt = PromptTemplate(
    template="""You are an expert document chunking agent. Analyze the text below and partition it into 
self-contained, coherent semantic chunks. Resolve any ambiguous pronouns so each chunk is fully standalone.

Formatting instructions: {format_instructions}

Document Text:
{text}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Example output schema returned by LLM:
mock_response = {
    "chunks": [
        {
            "chunk_title": "Quantum Computing Fundamentals",
            "chunk_content": "Quantum computers utilize qubits capable of superposition and entanglement, solving certain mathematical problems exponentially faster than classical computers."
        },
        {
            "chunk_title": "Cryogenic Hardware Requirements",
            "chunk_content": "Superconducting quantum processors require dilution refrigerators operating at near absolute zero temperatures (15 millikelvin) to prevent thermal decoherence."
        }
    ]
}
print(mock_response)
```

**Output Example:**
```json
{
  "chunks": [
    {
      "chunk_title": "Quantum Computing Fundamentals",
      "chunk_content": "Quantum computers utilize qubits capable of superposition and entanglement, solving certain mathematical problems exponentially faster than classical computers."
    },
    {
      "chunk_title": "Cryogenic Hardware Requirements",
      "chunk_content": "Superconducting quantum processors require dilution refrigerators operating at near absolute zero temperatures (15 millikelvin) to prevent thermal decoherence."
    }
  ]
}
```
</details>

---

#### 10. Hybrid chunking
Combines multiple chunking techniques sequentially in a multi-stage ingestion pipeline. For example:
1. **Stage 1 (Structure):** Split document into major sections using `MarkdownHeaderTextSplitter`.
2. **Stage 2 (Recursive / Token):** If any section exceeds max token length, split it using `RecursiveCharacterTextSplitter` or `TokenTextSplitter` with 15% overlap.
3. **Stage 3 (Metadata Propagation):** Preserve parent structural headers in the child chunks for filtered vector search.
* **Best used for:** Enterprise production RAG architectures requiring both structural awareness and strict token boundaries.
* **Risk:** Slightly more complex ingestion pipeline logic.

<details>
<summary><b>Code & Example: Hybrid Chunking</b></summary>

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

raw_document = """# System Architecture Guide

## Ingestion Engine
The ingestion engine extracts raw data from S3 buckets and databases. It handles decompression, OCR for PDFs, and character encoding sanitization. Once cleaned, text streams are prepared for downstream vectorization.

## Retrieval Engine
The retrieval engine combines dense vector search with BM25 keyword matching. It utilizes a reciprocal rank fusion algorithm to merge candidate sets before cross-encoder re-ranking.
"""

# Step 1: Structural Split by Headers
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Section"), ("##", "SubSection")],
    strip_headers=False
)
structural_chunks = header_splitter.split_text(raw_document)

# Step 2: Recursive Sub-splitting for token safety
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)
hybrid_chunks = text_splitter.split_documents(structural_chunks)

for i, doc in enumerate(hybrid_chunks, 1):
    print(f"Hybrid Chunk {i} | Metadata: {doc.metadata}")
    print(f"Content: {doc.page_content}\n")
```

**Output Example:**
```text
Hybrid Chunk 1 | Metadata: {'Section': 'System Architecture Guide', 'SubSection': 'Ingestion Engine'}
Content: ## Ingestion Engine
The ingestion engine extracts raw data from S3 buckets and databases.

Hybrid Chunk 2 | Metadata: {'Section': 'System Architecture Guide', 'SubSection': 'Ingestion Engine'}
Content: It handles decompression, OCR for PDFs, and character encoding sanitization. Once cleaned, text streams are prepared

Hybrid Chunk 3 | Metadata: {'Section': 'System Architecture Guide', 'SubSection': 'Retrieval Engine'}
Content: ## Retrieval Engine
The retrieval engine combines dense vector search with BM25 keyword matching.

Hybrid Chunk 4 | Metadata: {'Section': 'System Architecture Guide', 'SubSection': 'Retrieval Engine'}
Content: It utilizes a reciprocal rank fusion algorithm to merge candidate sets before cross-encoder re-ranking.
```
</details>

<br>

---

<br>

## 2. PDF Parsing (`2-dataparsingpdf.ipynb`) <a id="pdf-parsing" name="pdf-parsing"></a>
### Imports
```python
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
```
### How to Use
```python
# Load PDF page-by-page (PyMuPDF is fast and layout-accurate)
loader = PyMuPDFLoader("data/sample.pdf")
pages = loader.load()
```
### What They Do
*   `PyPDFLoader`: Simple, pure Python loader that extracts text page-by-page.
*   `PyMuPDFLoader`: C-based, high-accuracy PDF text extractor. <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Extremely fast (10x faster than PyPDF)</mark> and superior at multi-column layout parsing.

### 💡 Advanced Best Practices & Key Insights:
*   **Ligature & Whitespace Cleaning**: Raw PDF text often contains ligatures (`ﬁ`, `ﬂ`) or whitespace artifacts. Always run regex cleaning (`re.sub(r'\s+', ' ', text)`) post-ingestion.
*   **Scanned PDFs**: For image-based PDFs, standard loaders fail. Use OCR tools like `pdf2image` + `pytesseract` or `UnstructuredPDFLoader` with OCR strategy.

<br>

---

<br>

## 3. Word Document Parsing (`3-dataparsingdoc.ipynb`) <a id="word-parsing" name="word-parsing"></a>
### Imports
```python
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
from unstructured.partition.docx import partition_docx
```
### How to Use
```python
# Simple text extraction
loader = Docx2txtLoader("data/sample.docx")
docs = loader.load()

# Element-based structured extraction
unstructured_loader = UnstructuredWordDocumentLoader(
    "data/sample.docx", 
    mode="elements", 
    strategy="fast"
)
element_docs = unstructured_loader.load()
```
### What They Do
*   `Docx2txtLoader`: Fast, lightweight loader that extracts plain text from `.docx` files.
*   `UnstructuredWordDocumentLoader`: Partitions documents into granular logical elements (`Title`, `NarrativeText`, `Table`).

### 💡 Advanced Best Practices & Key Insights:
*   **Header/Footer Exclusion**: Use `UnstructuredWordDocumentLoader` with `mode="elements"` to filter out repeating header/footer noise from indexing.

<br>

---

<br>

## 4. CSV & Excel Structured Parsing (`4-csvexcelparsing.ipynb`) <a id="csv-excel-parsing" name="csv-excel-parsing"></a>
### Imports
```python
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader
import pandas as pd
```
### How to Use
```python
# Load CSV (each row is loaded as a separate Document object)
loader = CSVLoader("data/products.csv", source_column="Product")
docs = loader.load()
```
### What They Do
*   `CSVLoader`: Creates a Document object for each row of a CSV, writing columns as key-value text lines.
*   `UnstructuredExcelLoader`: Loads sheets and tables as text elements.
*   `pandas.DataFrame`: Used for custom CSV/Excel pre-processing before converting to LangChain Documents.

### 💡 Advanced Best Practices & Key Insights:
*   **Tabular Anti-Pattern**: Avoid indexing huge tables row-by-row into vector stores. Instead, format rows into rich natural language summaries (`"Product X belongs to Category Y with price $Z"`) for drastically better semantic retrieval.

<br>

---

<br>

## 5. JSON Parsing (`5-jsonparsing.ipynb`) <a id="json-parsing" name="json-parsing"></a>
### Imports
```python
from langchain_community.document_loaders import JSONLoader
```
### How to Use
```python
# Load specific values from nested JSON using jq path queries
loader = JSONLoader(
    "data/company.json", 
    jq_schema=".employees[].role", 
    text_content=True
)
docs = loader.load()
```
### What They Do
*   `JSONLoader`: Extracts specific JSON elements using `jq` path syntax (`.employees[].role`).

### 💡 Advanced Best Practices & Key Insights:
*   **JSON Noise Reduction**: Embed only human-readable descriptive values (`text_content=True`). Store structural JSON metadata (IDs, foreign keys) in the Document `metadata` dict for exact filtering.

<br>

---

<br>

## 6. Database Parsing (`6-databaseparsing.ipynb`) <a id="database-parsing" name="database-parsing"></a>
### Imports
```python
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import SQLDatabaseLoader
```
### How to Use
```python
# Connect to DB and load specific query outputs as Documents
db = SQLDatabase.from_uri("sqlite:///data/company.db")
loader = SQLDatabaseLoader(query="SELECT name, role FROM employees", db=db)
docs = loader.load()
```
### What They Do
*   `SQLDatabase`: Connects to relational databases (SQLite, PostgreSQL, MySQL) and exposes DDL schema metadata.
*   `SQLDatabaseLoader`: Executes queries and maps SQL result rows into LangChain Document objects.

### 💡 Advanced Best Practices & Key Insights:
*   **Security & Read-Only Access**: Never connect a Text-to-SQL or database ingestion agent using write/delete database credentials. Always scope SQL users to strictly <mark style="background-color: #f8d7da; color: #721c24; padding: 2px 4px; border-radius: 4px;">READ-ONLY</mark> permissions.

<br>

---

<br>

## 7. Embedding Models (`7.0-embedding.ipynb` & `7.1-openaiembeddings.ipynb`) <a id="embedding-models" name="embedding-models"></a>
### 🔑 Setting up API Keys for Cloud Embedding Models (e.g., OpenAI)

To use API-based embedding models like `OpenAIEmbeddings`, you need to set up your API key. There are two primary methods:

#### Method 1: Using `.env` File (Recommended Best Practice)
1. Install `python-dotenv`:
   ```bash
   pip install python-dotenv
   ```
2. Create a `.env` file in your root project directory:
   ```env
   OPENAI_API_KEY="your-actual-openai-api-key-here"
   ```
3. Load the environment variable in Python:
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()  # Automatically loads OPENAI_API_KEY into os.environ
   ```

#### Method 2: Passing directly in Constructor
```python
import os
from langchain_openai import OpenAIEmbeddings

openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="your-actual-openai-api-key-here" # Or os.getenv("OPENAI_API_KEY")
)
```

### Imports
```python
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()
```
### How to Use
```python
# Local HuggingFace Embeddings (No API Key Required - Runs Locally)
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_query = hf_embeddings.embed_query("your query text")

# API-Based OpenAI Embeddings (Requires OPENAI_API_KEY)
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_docs = openai_embeddings.embed_documents(["doc chunk 1", "doc chunk 2"])
```
### What They Do
*   `HuggingFaceEmbeddings`: Generates vector representations locally ($0 API cost, no API Key needed) using open-source models like `all-MiniLM-L6-v2`. Use `model_kwargs={'device': 'cuda'}` for GPU acceleration.
*   `OpenAIEmbeddings`: Generates high-quality semantic vectors via the OpenAI API (Requires `OPENAI_API_KEY`). Use the newer `text-embedding-3-small` model (cheaper and supports custom output dimension reduction).
*   **Key Methods**:
    *   `embed_documents(list_of_texts)`: Embeds multiple document chunks (indexing phase).
    *   `embed_query(single_text)`: Embeds the user query (search phase).
*   **Key Concept**: Cosine similarity measures the angle between vectors to check document similarity, bypassing issues with document length variance.

<br>

---

<br>

## 8. Vector Databases (`8.1` - `8.4`) <a id="vector-databases" name="vector-databases"></a>

### Overview & Comparison
Vector databases store and index high-dimensional vector embeddings generated by machine learning models to perform fast nearest-neighbor similarity searches (like Cosine Similarity or Euclidean L2 Distance).

| Vector DB | Type | Storage / Persistence | Key Advantage | Best Use Case |
|-----------|------|-----------------------|---------------|---------------|
| **ChromaDB** | <mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">Open Source / Local & Cloud</mark> | Disk (SQLite/Parquet) or Cloud Server / Hosted Cloud | <mark style="background-color: #e2e3e5; color: #383d41; padding: 2px 4px; border-radius: 4px;">Zero-config local setup</mark>, persistence, client/server & managed cloud support | <mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">Local RAG & rapid prototyping</mark>, microservices via Chroma Cloud |
| **FAISS** | <mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">In-Memory C++ Library</mark> | Local Files (`.index` / `.pkl`) | <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Blazing fast GPU/CPU vector search</mark> | <mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">In-memory batch vector searches</mark>, localized search without server overhead |
| **Pinecone** | <mark style="background-color: #f8d7da; color: #721c24; padding: 2px 4px; border-radius: 4px;">Cloud Managed Service</mark> | Fully Managed Serverless Cloud | <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Zero infra management</mark>, enterprise scaling & real-time updates | <mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">Production RAG pipelines</mark>, multi-tenant SaaS applications |
| **AstraDB / DataStax** | <mark style="background-color: #f8d7da; color: #721c24; padding: 2px 4px; border-radius: 4px;">Cloud Managed Service</mark> | Serverless Cassandra | <mark style="background-color: #e2e3e5; color: #383d41; padding: 2px 4px; border-radius: 4px;">Vector + NoSQL JSON docs</mark>, global scale & low latency | <mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">Enterprise hybrid data RAG</mark> requiring document storage |
| **Qdrant** | <mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">Open Source / Local & Cloud</mark> | Local RAM (`:memory:`), Disk (`./qdrant_db`), Docker, or Cloud Serverless | <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Rust performance, quantization (40x speedup), payload indexing</mark> | <mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">High-scale local & production RAG</mark>, advanced metadata filtering, low RAM footprint |
| **InMemoryVectorStore** | <mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">Core Store</mark> | In-Memory Python Dict | <mark style="background-color: #e2e3e5; color: #383d41; padding: 2px 4px; border-radius: 4px;">Zero external dependencies</mark>, pure Python execution | <mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">Unit testing & CI/CD</mark>, single-session demo scripts |

---

### 💡 When to Use Which Vector Database? (Decision Guide)

1. **Use <mark style="background-color: #fff3cd; color: #856404; padding: 2px 6px; border-radius: 4px;">ChromaDB</mark> when:**
   * You are building **local RAG applications**, Python scripts, or desktop tools.
   * You want an easy, lightweight database that runs embedded in Python or in Docker containers.
   * You plan to transition from local testing to a remote microservice or managed **<mark style="background-color: #d1ecf1; color: #0c5460; padding: 2px 4px; border-radius: 4px;">Chroma Cloud</mark>** (`HttpClient` / Hosted Service) without changing your application query logic.

2. **Use <mark style="background-color: #d4edda; color: #155724; padding: 2px 6px; border-radius: 4px;">FAISS</mark> when:**
   * You need **<mark style="background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px;">maximum similarity search speed</mark>** over fixed/static vector datasets.
   * You want to run searches purely in memory or perform high-throughput **GPU-accelerated** indexing.
   * You do **not** need multi-user concurrency, API server features, or real-time CRUD operations.

3. **Use <mark style="background-color: #f8d7da; color: #721c24; padding: 2px 6px; border-radius: 4px;">Pinecone</mark> when:**
   * You are deploying **<mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">production RAG applications</mark>** to cloud environments.
   * You require a **fully managed serverless infrastructure** with automatic scaling, backup, high availability, and metadata filtering.
   * You want zero infrastructure maintenance (no managing servers or disk storage).

4. **Use <mark style="background-color: #e2e3e5; color: #383d41; padding: 2px 6px; border-radius: 4px;">DataStax / AstraDB</mark> when:**
   * You require enterprise-grade serverless cloud vector search backed by Apache Cassandra.
   * You need both **rich JSON document/NoSQL storage** alongside vector embeddings.

5. **Use <mark style="background-color: #d4edda; color: #155724; padding: 2px 6px; border-radius: 4px;">Qdrant</mark> when:**
   * You need **Rust-powered ultra-fast vector search** with seamless portability from local prototyping (`path="./qdrant_db"`) to self-hosted Docker or managed **Qdrant Cloud**.
   * You need **memory compression (Vector Quantization)** to store millions/billions of vectors in RAM with up to 95% memory savings and 40x speedups.
   * You require **rich boolean payload filtering** (Must, Should, Must Not) and hybrid dense + sparse keyword search.

6. **Use <mark style="background-color: #fff3cd; color: #856404; padding: 2px 6px; border-radius: 4px;">InMemoryVectorStore</mark> when:**
   * You are running **unit tests**, continuous integration (CI) tests, or quick 1-file proof-of-concepts where vectors don't need to persist after Python exits.

---

### 1. Chroma (`langchain_chroma.Chroma`) <a id="chroma-db" name="chroma-db"></a>
Chroma is an open-source, developer-friendly vector database. It supports 3 deployment modes:
* 🟢 **Embedded Mode**: Runs inside your Python process and persists data to disk via SQLite/Parquet (ideal for local development).
* 🟡 **Client/Server Mode**: Runs Chroma as a standalone Docker container or separate microservice (`HttpClient`).
* 🔵 **Chroma Cloud**: Fully managed serverless cloud service (`chromadb.CloudClient`) without local infrastructure overhead.

#### Imports
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
```

#### Complete Implementation (Create, Persist, Load & Query)
```python
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class ChromaVectorStoreManager:
    def __init__(self, persitent_dir="./chroma_db"):
        self.persitent_dir = persitent_dir
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small") # 🔹 Standardize embedding model

    def create_and_persist_vector_store(self, chunks):
        """
        Creates a Chroma vector store from document chunks and automatically persists it to disk.
        """
        # 🔹 from_documents() builds vector index and writes directly to disk
        db = Chroma.from_documents(
            chunks,
            self.embedding,
            persist_directory=self.persitent_dir
        )
        return db

    def load_vector_store(self):
        """
        Loads an existing persisted Chroma vector store from disk.
        """
        return Chroma(
            persist_directory=self.persitent_dir,
            embedding_function=self.embedding
        )

# ⚡ Execution Flow:
# 1. Create store  -> manager.create_and_persist_vector_store(chunks)
# 2. Reload store  -> loaded_db = manager.load_vector_store()
# 3. Vector search -> loaded_db.similarity_search("What is RAG?", k=3)
```

> [!NOTE]
> **Is `db.persist()` still needed?**
> **No.** `db.persist()` is **deprecated and removed** in `chromadb` (v0.4.0+) & `langchain-chroma`. Data is saved automatically whenever `persist_directory` is specified. Calling `db.persist()` will throw an `AttributeError`.

#### ➕ Adding Data to Existing Vector Store (Incremental Ingestion)
To add new documents or raw text to an existing collection without rebuilding the database:

```python
from langchain_core.documents import Document

# 1. Load existing vector store
db = manager.load_vector_store()

# Option A: Adding LangChain Document objects (with metadata)
new_docs = [
    Document(page_content="New chunk content 1", metadata={"source": "news_api", "category": "tech"}),
    Document(page_content="New chunk content 2", metadata={"source": "news_api", "category": "finance"})
]
added_ids = db.add_documents(new_docs)  # Embeds and auto-persists

# Option B: Adding raw strings directly
added_ids = db.add_texts(
    texts=["Raw string 1 to embed", "Raw string 2 to embed"],
    metadatas=[{"author": "Alice"}, {"author": "Bob"}]
)
```

#### Metadata Pre-filtering & CRUD Operations
```python
# 1. Metadata Pre-Filtering Search
filtered_docs = db.similarity_search(
    "What is deep learning?",
    k=3,
    filter={"source": "data/sample.pdf", "page": 1}
)

# 2. Collection Inspection (Total Count)
print(f"Total stored vectors: {db._collection.count()}")

# 3. Delete Documents by ID
db.delete(ids=["doc_id_1", "doc_id_2"])
```

#### Core Methods Summary
* `Chroma.from_documents()`: Embeds chunks and auto-saves indexed vectors to `persist_directory`.
* `Chroma()`: Loads an existing persisted store from disk without re-embedding.
* `db.add_documents()` / `db.add_texts()`: Incrementally adds and auto-saves new data.
* `db.similarity_search()`: Performs semantic nearest-neighbor similarity search.

---

### 2. FAISS (`langchain_community.vectorstores.FAISS`) <a id="faiss" name="faiss"></a>
FAISS (Facebook AI Similarity Search) is a high-performance C++ library with Python bindings designed for fast similarity search and vector clustering.

#### Imports
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
```

#### Complete Implementation (Create, Save & Load)
```python
class FAISSVectorStoreManager:
    def __init__(self, persitent_dir="faiss_index"):
        self.persitent_dir = persitent_dir
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    def create_and_persist_vector_store(self, chunks):
        """
        Creates an in-memory FAISS index and writes files (.faiss + .pkl) to disk.
        """
        db = FAISS.from_documents(chunks, self.embedding)
        
        # 🔹 MANDATORY: Save binary vector index & metadata docstore to disk
        db.save_local(folder_path=self.persitent_dir)
        return db

    def load_vector_store(self):
        """
        Loads a saved FAISS index from disk into memory.
        """
        return FAISS.load_local(
            folder_path=self.persitent_dir,
            embeddings=self.embedding,
            allow_dangerous_deserialization=True  # Required to safely unpickle docstore
        )
```

> [!WARNING]
> **Does Auto-Save happen in FAISS like Chroma?**
> **NO! FAISS is purely IN-MEMORY.**
> - Chroma auto-writes to SQLite on disk, while FAISS holds vectors strictly in Python RAM.
> - **Mandatory**: You **MUST** call `db.save_local(folder_path)` after creating or updating (`add_documents()`) a FAISS store to persist changes.
> - Exiting Python without calling `save_local()` will lose all newly added vectors!

#### ➕ Adding Data & Saving (FAISS Incremental Workflow)
```python
# 1. Load FAISS store into memory
db = manager.load_vector_store()

# 2. Add new documents into RAM
db.add_documents(new_docs)

# 3. Save updated RAM state back to disk
db.save_local(folder_path="faiss_index")
```

#### Core Methods Summary
* `FAISS.from_documents()`: Constructs an in-memory vector index.
* `db.save_local()`: Serializes and saves index binary (`index.faiss`) & metadata (`index.pkl`) to disk.
* `FAISS.load_local()`: Loads saved index files back into RAM.
* `db.add_documents()`: Appends new vectors to RAM (must be followed by `db.save_local()`).

---

### 3. Pinecone (`langchain_pinecone.PineconeVectorStore`) <a id="pinecone" name="pinecone"></a>
Pinecone is a cloud-native, fully-managed serverless vector database designed for production scaling, high availability, and metadata pre-filtering.

#### Imports
```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
```

#### Complete Implementation (Create & Load)
```python
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

class PineconeVectorStoreManager:
    def __init__(self, index_name="rag-index"):
        self.index_name = index_name
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    def create_and_persist_vector_store(self, chunks):
        """
        Creates index if missing and embeds chunks directly into cloud Pinecone index.
        """
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # Vector dimensions for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
        # Create and persist chunks to cloud vector store
        db = PineconeVectorStore.from_documents(
            chunks,
            self.embedding,
            index_name=self.index_name
        )
        return db

    def load_vector_store(self):
        """
        Connects to an existing cloud-hosted Pinecone vector store index.
        """
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embedding
        )
```

---

### 4. InMemoryVectorStore (`langchain_core.vectorstores.InMemoryVectorStore`) <a id="inmemory-store" name="inmemory-store"></a>
`InMemoryVectorStore` is the simplest zero-dependency transient vector store provided natively by `langchain-core` (LangChain 0.2+ standard). It stores vectors in a plain Python dictionary in memory (RAM).

#### Imports
```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
```

#### Basic Usage (In-Memory Execution)
```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create store in RAM
db = InMemoryVectorStore.from_documents(chunks, embeddings)

# Query store
results = db.similarity_search("Explain vector embeddings", k=2)
```

#### 💾 Can you Save & Load `InMemoryVectorStore` to/from Disk?
By default, `InMemoryVectorStore` data is lost when Python exits. However, because it is a pure Python object, **you CAN serialize it to disk** (save) and reload it later using Python's `pickle` or custom JSON dumping!

##### Save & Load using `pickle` (File Dump):
```python
import pickle

# 1. Save (Dump) in-memory vector store to disk
with open("in_memory_store.pkl", "wb") as f:
    pickle.dump(db, f)

# 2. Load (Restore) in-memory vector store from disk
with open("in_memory_store.pkl", "rb") as f:
    loaded_db = pickle.load(f)

# Query loaded store
results = loaded_db.similarity_search("What is RAG?", k=2)
```

> [!TIP]
> **When to use `InMemoryVectorStore`?**
> Ideal for **unit testing (CI/CD)**, short interactive demo scripts, or single session apps where you don't want external database binaries installed on your system.

---

### 5. Qdrant (`langchain_qdrant.QdrantVectorStore` & `qdrant_client.QdrantClient`) <a id="qdrant" name="qdrant"></a>
Qdrant is an enterprise-grade open-source vector search engine and database written in Rust. It offers extreme vector search speed, rich payload (metadata) filtering, memory quantization (up to 95% RAM compression), and hybrid dense + sparse retrieval.

Qdrant supports **4 flexible execution modes**:
* 🟢 **Local In-Memory Mode (`location=":memory:"`)**: Runs purely in RAM with zero disk writes (ideal for tests and one-off scripts).
* 🟡 **Local Disk Persistence (`path="./qdrant_db"`)**: Persists vectors and metadata directly to local disk without Docker or servers.
* 🟠 **Local Docker Container / Self-Hosted (`url="http://localhost:6333"`)**: Standalone server with built-in Web UI Dashboard (`http://localhost:6333/dashboard`).
* 🔵 **Qdrant Cloud Serverless (`url="https://<cluster-id>.qdrant.tech:6333"`, `api_key="<api-key>"`)**: Fully managed cloud service for high-concurrency production workloads.

#### Installation
```bash
pip install -qU qdrant-client langchain-qdrant langchain-openai langchain-core
```

#### Imports
```python
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
```

#### Complete Implementation (Local & Cloud Modes)
```python
class QdrantVectorStoreManager:
    def __init__(self, mode="local_disk", path="./qdrant_db", url=None, api_key=None, collection_name="rag_knowledge_base"):
        self.collection_name = collection_name
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.embedding_dim = 1536
        
        # 🔹 Initialize Qdrant Client based on desired deployment mode
        if mode == "memory":
            self.client = QdrantClient(location=":memory:")
        elif mode == "local_disk":
            self.client = QdrantClient(path=path)
        elif mode == "docker":
            self.client = QdrantClient(url=url or "http://localhost:6333")
        elif mode == "cloud":
            self.client = QdrantClient(
                url=url or os.getenv("QDRANT_CLOUD_URL"),
                api_key=api_key or os.getenv("QDRANT_API_KEY")
            )

    def create_and_persist_vector_store(self, chunks):
        """
        Ensures collection exists with cosine metric and ingests document chunks.
        """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )
        
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding
        )
        vector_store.add_documents(chunks)
        return vector_store

    def load_vector_store(self):
        """
        Connects to an existing Qdrant collection.
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding
        )

# ⚡ Usage Example:
# 1. Local Disk: manager = QdrantVectorStoreManager(mode="local_disk", path="./qdrant_db")
# 2. Cloud Mode: manager = QdrantVectorStoreManager(mode="cloud", url="https://xxxx.qdrant.tech:6333", api_key="...")
# 3. Create Store: db = manager.create_and_persist_vector_store(chunks)
```

#### ➕ Adding Data to Existing Qdrant Collection (Incremental Ingestion)
```python
# 1. Load existing Qdrant store
db = manager.load_vector_store()

# Option A: Adding LangChain Document chunks (with metadata)
new_docs = [
    Document(page_content="Scalar Quantization compresses 32-bit float vectors into 8-bit integers.", metadata={"topic": "Quantization", "author": "dev"}),
    Document(page_content="Binary Quantization yields up to 40x speedup and 95% RAM compression.", metadata={"topic": "Quantization", "author": "dev"})
]
db.add_documents(new_docs)

# Option B: Adding raw strings directly
db.add_texts(
    texts=["Qdrant supports exact payload index structures for instant metadata filtering."],
    metadatas=[{"topic": "Indexing", "author": "architect"}]
)
```

#### 🎯 Metadata Pre-filtering & Boolean Queries
```python
# Option 1: Simple dictionary filter
filtered_docs = db.similarity_search(
    "How does quantization work?",
    k=2,
    filter={"topic": "Quantization"}
)

# Option 2: Advanced Native Qdrant Filter (Must / Should / Must-Not boolean expressions)
qdrant_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="metadata.topic",
            match=models.MatchValue(value="Quantization")
        )
    ]
)
advanced_filtered = db.similarity_search("memory optimization", k=2, filter=qdrant_filter)
```

#### 🚀 Converting to Retriever & LCEL RAG Chain
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

# 1. Convert Qdrant to Retriever with Maximal Marginal Relevance (MMR)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
)

# 2. Build LCEL RAG Pipeline
prompt = ChatPromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}\nAnswer:")
llm = init_chat_model("gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("Explain quantization in Qdrant"))
```

#### Core Methods Summary
* `QdrantClient(path=...)` / `QdrantClient(url=..., api_key=...)`: Initializes local disk, memory, Docker, or remote cloud client.
* `client.create_collection()`: Configures vector dimension and distance metric (`Distance.COSINE`, `Distance.DOT`, `Distance.EUCLID`).
* `QdrantVectorStore()`: Connects LangChain to the Qdrant collection.
* `db.add_documents()` / `db.add_texts()`: Dynamically ingests and indexes vectors + payloads.
* `db.as_retriever()`: Exposes standard LangChain retrieval interface with `similarity`, `mmr`, or `similarity_score_threshold`.

---

### Understanding Vector Distance Metrics & Similarity Scores <a id="distance-metrics" name="distance-metrics"></a>

Similarity search relies on mathematical distance metrics between high-dimensional vector embeddings:

1. **Cosine Similarity**:
   * Measures the angle cosine between two vectors.
   * **Range**: `-1.0` to `1.0` (or normalized `0.0` to `1.0`).
   * **Interpretation**: **Higher is MORE similar**. `1.0` represents identical vector direction regardless of text length.

2. **Euclidean Distance (L2 Distance)**:
   * Measures straight-line geometric distance between vector points in multi-dimensional space.
   * **Range**: `0.0` to `+∞`.
   * **Interpretation**: **Lower is MORE similar**. `0.0` represents identical vectors.

3. **Dot Product (Inner Product)**:
   * Measures both vector angle and magnitude. Fast for normalized vectors where dot product equals cosine similarity.

> **Important Note on Score Sorting**:
> * `db.similarity_search(query)` returns Documents ordered by relevance.
> * `db.similarity_search_with_score(query)` returns tuples `(Document, score)`. When using L2 distance metrics (e.g., Chroma default), **smaller scores represent closer matches**. When using cosine similarity, **larger scores represent closer matches** prints.

---

## 9. RAG Chains & Conversational Memory (`8.1-chromadb.ipynb`) <a id="rag-chains" name="rag-chains"></a>

![RAG Architecture](assets/image-2.png)

### Imports
```python
# LLM Initialization
from langchain_openai import ChatOpenAI
from langchain.chat_models.base import init_chat_model

# Prompts & Message Structure
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Output Parsers & Runnables (LCEL)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Chains & Retrievers (Built-in - langchain_classic for LangChain 1.x compatibility)
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
```

### How to Use

#### 1. LLM / Model Initialization Methods <a id="llm-init-methods" name="llm-init-methods"></a>

**Method A: Using LangChain's Factory Function**
```python
# Uses environment variables (OPENAI_API_KEY, OPENAI_BASE_URL) automatically
llm = init_chat_model("openai:gpt-4.1-mini") 
```

**Method B: Making a Model Using Native Client (OpenAI API directly)**
If you want to bypass LangChain and interact with the LLM directly using a client:
```python
from openai import OpenAI

# Initialize the client
client = OpenAI(
    api_key="your_api_key",
    base_url="https://api.euron.one/api/v1/euri" # Optional: for custom providers
)

# Make a model request using the client
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "What is Deep Learning?"}]
)
print(response.choices[0].message.content)
```

#### 2. Custom RAG Chain using LCEL (LangChain Expression Language) <a id="lcel-rag-chain" name="lcel-rag-chain"></a>

**LCEL (LangChain Expression Language)** is a declarative way to compose and chain artificial intelligence building blocks—such as prompts, models, and parsers—using the pipe operator (`|`). [[1](https://www.geeksforgeeks.org/artificial-intelligence/langchain/), [2](https://www.langchain.com/blog/langchain-expression-language)]

##### 🧠 What is LCEL?
* **Declarative Composition:** You define what components to connect, and data flows automatically from left to right.
* **The Runnable Protocol:** Every core element in LCEL implements a standard interface (Runnables) that handles execution seamlessly.
* **Basic Syntax:** A standard workflow looks like `chain = prompt | llm | output_parser`. [[1](https://cobusgreyling.medium.com/what-is-langchain-expression-language-lcel-8a828c38b37d), [2](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/01-basic/07-lcel-interface), [3](https://www.aurelio.ai/learn/langchain-lcel), [4](https://www.geeksforgeeks.org/artificial-intelligence/langchain/)]

##### 🚀 Key Features & Benefits
* **Out-of-the-Box Execution Modes:** Supports synchronous (`invoke`), asynchronous (`ainvoke`), batch (`batch`), and streaming (`stream`) execution without changing your code. [[1](https://www.youtube.com/watch?v=8aUYzb1aYDU&t=1), [2](https://k21academy.com/ai-ml/langchain-expression-language/), [3](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/01-basic/07-lcel-interface)]
* **Automatic Parallelism:** Steps that can run concurrently do so automatically to boost runtime efficiency. [[1](https://k21academy.com/ai-ml/langchain-expression-language/)]
* **Production Ready:** Designed to transition smoothly from local prototypes to production environments with built-in logging and tracing via platforms like LangSmith. [[1](https://www.artefact.com/blog/unleashing-the-power-of-langchain-expression-language-lcel-from-proof-of-concept-to-production/), [2](https://www.langchain.com/blog/langchain-expression-language), [3](https://k21academy.com/ai-ml/langchain-expression-language/)]

```python
# Initialize LLM
llm = init_chat_model("openai:gpt-4.1-mini") 

# Define a custom prompt template
custom_prompt = ChatPromptTemplate.from_template("""Use the following context to answer the question. 
If you don't know the answer based on the context, say you don't know.
Provide specific details from the context to support your answer.

Context:
{context}
                                                 
Question: {question}

Answer: """)

# Setup retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the custom RAG Chain using LCEL
rag_chain_lcel = (
    {
        "context": retriever | format_docs, 
        "question": RunnablePassthrough()
    }
    | custom_prompt
    | llm
    | StrOutputParser()
)

# Test/Invoke the chain
response = rag_chain_lcel.invoke("What is Deep Learning")

# Query function using the LCEL approach
def query_rag_lcel(question):
    print(f"Question: {question}")
    print("-" * 50)
    
    # Pass string query directly to the chain
    answer = rag_chain_lcel.invoke(question)
    print(f"Answer: {answer}")
    
    # Get source documents separately for inspection
    docs = retriever.invoke(question)
    print("\nSource Documents:")
    for i, doc in enumerate(docs):
        print(f"\n--- Source {i+1} ---")
        print(doc.page_content[:200] + "...")

# Run verification
query_rag_lcel("What are the key concepts in reinforcement learning?")
```

#### 3. Conversational RAG Chain (With History/Memory) <a id="conversational-rag" name="conversational-rag"></a>

##### Imports
```python
# Document Retrieval & Chain Construction
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Local Vector Database & Embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Prompts & Messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
```

##### Full Execution Code
```python
# Initialize empty chat history list
chat_history = []

# 1. Setup Vector Store Retriever & LLM
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# 2. Contextualize Question Prompt (Reformulates question using history context)
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 3. Create History-Aware Retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 4. Answer Generation Prompt
qa_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 5. Build Complete Conversational RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever, 
    question_answer_chain
)

# Turn 1: Initial Question
result1 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What is machine learning?"
})
print(f"Q: What is machine learning?")
print(f"A: {result1['answer']}\n")

# Update history
chat_history.extend([
    HumanMessage(content="What is machine learning?"),
    AIMessage(content=result1['answer'])
])

# Turn 2: Follow-up question (refers to ML from previous question)
result2 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What are its main types?"
})
print(f"Q: What are its main types?")
print(f"A: {result2['answer']}")
```

#### 4. Modern RAG Chain (Using LangChain Classic Retrieval Chain) <a id="classic-rag-chain" name="classic-rag-chain"></a>
![alt text](image-1.png)
The classic RAG chain uses helper functions like `create_stuff_documents_chain` and `create_retrieval_chain` to quickly stitch together a retriever, prompt, and LLM.

```python
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Configure Environment (Euron API compatibility)
os.environ["OPENAI_API_KEY"] = os.getenv("EURI_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://api.euron.one/api/v1/euri"

sample_text = "Machine Learning is fascinating"
embeddings = OpenAIEmbeddings()

persist_directory = "./chroma_db"

# Create a ChromaDB vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="rag_collection"
)

print(f"Vector store named : {vectorstore._collection.name} ")
print(f"Vector store created with {vectorstore._collection.count()} vectors")
print(f"Persisted to: {persist_directory}")

# Setup retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Setup prompt template
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create stuff documents chain & retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Invoke the RAG chain
response = rag_chain.invoke({"input": "What is Deep Learning"})
```

#### 5. When to Use vs. When NOT to Use `format_docs` in LangChain <a id="format-docs-guide" name="format-docs-guide"></a>

In LangChain, deciding whether you need a `format_docs` helper depends entirely on **how your chain is constructed**:

---

##### 1. **When to USE `format_docs`** 
👉 **When building custom LCEL (LangChain Expression Language) chains directly.**

```python
# Pure LCEL Pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

###### Why it's needed here:
- `retriever` returns a Python list of `Document` objects (`List[Document]`).
- A standard `ChatPromptTemplate` expects a **string** for `{context}`.
- If you pass `List[Document]` directly without `format_docs`, the prompt will receive the raw Python object representation (e.g. `[Document(page_content='...'), ...]`), wasting tokens and confusing the LLM.
- **You also use `format_docs` when you want custom formatting**, such as injecting metadata/source attribution into the context:
  ```python
  def format_docs_with_sources(docs):
      return "\n\n".join(
          f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}"
          for doc in docs
      )
  ```

---

##### 2. **When NOT to use `format_docs`**
👉 **When using LangChain’s pre-built helper chains like `create_stuff_documents_chain` and `create_retrieval_chain`.**

```python
# Built-in LangChain Helpers
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_conversational_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

###### Why you don't need it here:
- `create_stuff_documents_chain` is built specifically to accept `List[Document]` as its input.
- **It formats documents internally** using its default document template (`{page_content}`) and joins them with `\n\n`.
- `create_retrieval_chain` passes the raw `docs` into `create_stuff_documents_chain`, and also preserves the original `List[Document]` in the final output dictionary (`response["context"]`), allowing you to inspect sources, scores, or metadata later.
- If you manually pass a pre-formatted string instead of `List[Document]` to `create_stuff_documents_chain`, it will fail because it expects document objects.

---

##### Quick Comparison Summary

| Feature | LCEL Chain (`retriever \| format_docs \| prompt`) | Pre-built Chain (`create_stuff_documents_chain`) |
| :--- | :--- | :--- |
| **`format_docs` required?** | **Yes** (Must convert `List[Document]` $\rightarrow$ `str`) | **No** (Handles formatting internally) |
| **Input to `{context}` in prompt** | Plain String | Raw `List[Document]` handled under the hood |
| **Final Output** | Typically just the string response | Dictionary containing `answer` + raw `context` docs |
| **Custom formatting** | Handled in your Python function | Configured via `document_prompt` & `document_separator` |
| **Best suited for** | Lightweight, fully customized, streaming LCEL pipelines | Standard RAG, multi-turn chat history, and source tracking |

---

### What They Do
*   `ChatOpenAI` / `init_chat_model`: Direct instantiation vs a configurable factory helper to initialize chat models.
*   `ChatPromptTemplate`: Creates structured message prompts for the LLM.
*   `StrOutputParser`: Extracts the string content from the LLM's response message object.
*   `RunnablePassthrough`: Passes the input unmodified through the current step (useful for mapping user queries).
*   `format_docs`: Custom helper function to transform a `List[Document]` into a clean concatenated string for raw LCEL prompt context injection.
*   `create_stuff_documents_chain`: Combines a list of documents into a single prompt template context window.
*   `create_retrieval_chain`: Chains a retriever and stuff-documents chain together.
*   `create_history_aware_retriever`: Combines conversation history and user query, asking the LLM to draft a standalone query *before* searching the Vector DB. Ensures correct pronoun resolution (e.g. "it", "them").

---

## 10. Semantic Chunking (`9.1-semantichunking.ipynb`) <a id="semantic-chunking" name="semantic-chunking"></a>

### Imports
```python
# Pure Python / SentenceTransformers implementation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# LangChain & RAG Pipeline Integration
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

### How to Use

#### Method A: Custom Threshold-Based Semantic Chunker (Pure Python / SentenceTransformers)
```python
class ThresholdSemanticChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.7):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold 

    def split(self, text: str):
        # 1. Split text into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return []
        
        # 2. Compute embeddings for all sentences
        embeddings = self.model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]

        # 3. Calculate cosine similarity between consecutive adjacent sentences
        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if sim >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentences[i]]

        chunks.append(". ".join(current_chunk) + ".")
        return chunks

    def split_documents(self, docs):
        result = []
        for doc in docs:
            for chunk in self.split(doc.page_content):
                result.append(Document(page_content=chunk, metadata=doc.metadata))
        return result

# Usage
chunker = ThresholdSemanticChunker(threshold=0.7)
chunks = chunker.split_documents([Document(page_content="...")])
```

#### Method B: Built-in LangChain `SemanticChunker`
```python
# 1. Load Document
loader = TextLoader("langchain_intro.txt")
docs = loader.load()

# 2. Initialize embedding model (e.g., OpenAI or local HuggingFace)
embeddings = OpenAIEmbeddings()

# 3. Initialize Semantic Chunker (default breakpoint threshold type: percentile)
chunker = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="percentile" # Options: 'percentile', 'standard_deviation', 'interquartile', 'gradient'
)

# 4. Split documents into semantically coherent chunks
semantic_chunks = chunker.split_documents(docs)
```

#### Method C: Modular RAG Pipeline with Semantic Chunker & FAISS
```python
# 1. Custom semantic chunking
chunker = ThresholdSemanticChunker(threshold=0.7)
chunks = chunker.split_documents(docs)

# 2. Store in FAISS Vector Store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# 3. Prompt & LLM Setup
prompt = PromptTemplate.from_template(
    "Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}\n"
)
llm = init_chat_model(model="groq:gemma2-9b-it", temperature=0.4)

# 4. Build LCEL Chain
rag_chain = (
    RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Run Query
result = rag_chain.invoke({"question": "What is LangChain used for?"})
print(result)
```

### What They Do
*   `SemanticChunker`: Splits documents dynamically based on sentence embedding similarity rather than arbitrary length-based character limits.
*   `cosine_similarity`: Measures cosine similarity between consecutive sentence vectors. If similarity falls below `threshold`, a topic shift is detected and a chunk boundary is placed.
*   `breakpoint_threshold_type`: Strategy used by LangChain to calculate split thresholds:
    *   `percentile` (default): Splits when sentence distance exceeds a specified percentile cutoff across the document.
    *   `standard_deviation`: Splits based on standard deviation distance from the mean similarity score.
    *   `interquartile`: Uses interquartile range (IQR) to detect statistical anomalies/topic shifts.
    *   `gradient`: Looks for peaks in cosine distance gradients across sentences.

### 💡 Interview & Learning Notes

#### **Key Interview Questions:**
1. **Why use Semantic Chunking instead of `RecursiveCharacterTextSplitter`?**
   * Fixed-size chunkers (like character/token splitters) often cut text in the middle of a paragraph or thought, separating pronouns from their subjects.
   * Semantic chunking calculates similarity between adjacent sentences and only splits when similarity drops below a threshold (signaling a topic change). This preserves semantic context and leads to higher RAG retrieval quality.

2. **What is the main downside of Semantic Chunking?**
   * **Cost & Time**: Requires running an embedding model on *every single sentence* during the ingestion phase before final chunks are created. Using commercial API embeddings (e.g., OpenAI) for chunking can be expensive and slow.

#### **Learning Takeaways:**
* Use a fast, free local embedding model (e.g., `all-MiniLM-L6-v2`) for the sentence-level chunking pass, and save API embeddings (e.g., OpenAI) for indexing into the final vector database.

### 🚀 Best Practices for Semantic Chunking

1. **Cost & Token Optimization**:
   * Never use expensive API models (`text-embedding-3-large`) inside `SemanticChunker`. Use a fast local model (`HuggingFaceEmbeddings` / `SentenceTransformer`) for chunking logic, and use OpenAI embeddings only during final `vectorstore` indexing.

2. **Speed & Scalability**:
   * Semantic chunking is computationally heavy. For massive datasets (millions of documents), use standard `RecursiveCharacterTextSplitter` unless context fragmentation is demonstrably degrading answer quality.

3. **Threshold Tuning**:
   * The `percentile` breakpoint threshold metric is typically the most robust across varying document types and lengths.

---

## RAG Chain Types Comparison <a id="chain-types-comparison" name="chain-types-comparison"></a>

| Chain Type | Description | Key Features | Typical Use Cases |
|------------|-------------|--------------|-------------------|
| **Normal RAG Chain** | Retrieve a fixed set of top‑k documents, concatenate them, and pass the whole text to the LLM in a single prompt. | • Simple to implement<br>• No state across turns<br>• Limited by LLM context window | • One‑off Q&A<br>• Fact‑lookup where the answer fits in a single request |
| **Conversational RAG Chain** | Extends the normal chain with a memory component (chat history) and a history‑aware retriever that rewrites the user query using the prior conversation. | • Maintains multi‑turn context<br>• Resolves pronouns and references<br>• Uses `create_history_aware_retriever` and `MessagesPlaceholder` | • Customer‑support bots<br>• Interactive tutoring with follow‑up questions |
| **Streaming RAG Chain** | Retrieves documents incrementally and streams them to the LLM as they become available (e.g., using LangChain’s Runnable streaming or async generators). | • Handles very large corpora beyond the LLM token limit<br>• Overlaps retrieval and generation to reduce latency<br>• Can provide partial answers early | • Long‑form summarisation<br>• Real‑time assistance over massive knowledge bases |

---

## Vector Store vs Vector Database <a id="vector-store-vs-db" name="vector-store-vs-db"></a>

| Type | Persistence | Scaling | Metadata / Filtering | Typical Scenarios |
|------|--------------|---------|----------------------|-------------------|
| **In‑Memory Vector Store** (`InMemoryVectorStore`) | Pure Python dict, lives only while the process runs | Limited to a single process, RAM‑bound | Minimal – usually only the vector itself | Unit tests, quick prototypes, notebooks |
| **Local Vector Store** (`FAISS`, `Chroma`) | Files on disk (`.index`, SQLite for Chroma) | Works on a single machine; can handle millions of vectors with appropriate hardware | Supports basic metadata filters (e.g., `metadata['source'] == 'pdf'`) | Personal projects, research notebooks, small‑scale apps |
| **Managed Vector Database** (`Pinecone`, `Weaviate`, `Qdrant`, `Milvus Cloud`) | Cloud‑managed storage with replication & backups | Horizontally scalable, multi‑region, handles billions of vectors | Rich metadata queries, hybrid search (vector + scalar), security controls | Production SaaS products, multi‑tenant services, real‑time recommendation engines |

*When to choose which:* 
- **Prototype / Experiment** → start with **FAISS** or **Chroma** for speed and zero‑cost.
- **Production with modest load** → **Chroma** persisted locally or a lightweight **Qdrant** instance.
- **Enterprise‑grade, high‑throughput** → a managed service like **Pinecone** or **Weaviate** that offers SLA, automatic scaling, and fine‑grained metadata filtering.


---

## 11. Hybrid Search & Re-ranking (`05_hybrid search`) <a id="hybrid-search" name="hybrid-search"></a>

### 🧠 Section Overview: Why Do We Need Hybrid Search & Re-ranking?
Standard vector search (Dense Retrieval) relies purely on embedding distances (e.g. Cosine similarity). However:
- Vector search often fails on **exact keyword queries**, technical IDs, acronyms, and proper nouns (e.g., searching `"SKU-9921"` or `"Error 404"`).
- Vector search often returns **duplicate or near-identical text chunks** that fill up the LLM context window without adding new information.
- Vector search can return **false-positive semantically similar chunks** that don't actually answer the prompt.

**Solution**: We use **Hybrid Search** (Dense + Sparse BM25), **MMR** (Relevance + Diversity), and **Re-ranking** (2-Stage Cross-Encoder scoring) to achieve production-grade precision and recall.

---

### 11.1 Hybrid Retriever – Dense & Sparse Combination (`1-densesparse.ipynb`) <a id="dense-sparse-retriever" name="dense-sparse-retriever"></a>

#### 🎯 Why We Need This:
Neither vector search nor keyword search is perfect on its own. Hybrid retrieval combines both to get the "best of both worlds":
* **Dense Retriever (Vector Embeddings)**: Captures conceptual meaning, intent, synonyms, and context matching.
* **Sparse Retriever (BM25 Keyword Search)**: Captures exact word matches, codes, names, and rare technical terminology.
* **EnsembleRetriever (RRF)**: Merges scores from both retrievers using Reciprocal Rank Fusion (RRF).

#### Imports
```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
```

#### How to Use
```python
# 1. Sample documents & Dense Retriever (FAISS + HuggingFace)
docs = [
    Document(page_content="LangChain helps build LLM applications."),
    Document(page_content="Pinecone is a vector database for semantic search."),
    Document(page_content="LangChain can be used to develop agentic AI applications.")
]
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
dense_vectorstore = FAISS.from_documents(docs, embedding_model)
dense_retriever = dense_vectorstore.as_retriever()

# 2. Sparse Retriever (BM25)
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 3

# 3. Combine using EnsembleRetriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]
)

# 4. Invoke hybrid search or connect to RAG chain
results = hybrid_retriever.invoke("How can I build an application using LLMs?")
```

#### What They Do
*   `Dense Retriever`: Uses vector embeddings to capture semantic context, handling synonyms and concept matching.
*   `Sparse Retriever` (`BM25`): Uses keyword-based frequency scoring (TF-IDF family) to catch exact term matches, proper nouns, and technical IDs.
*   `EnsembleRetriever`: Combines search results from multiple retrievers using Reciprocal Rank Fusion (RRF) with user-defined weights (`weights=[0.7, 0.3]`).
*   **Reciprocal Rank Fusion (RRF)**: Re-ranks items by summing reciprocal ranks across retrievers using formula $RRF\_Score(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$ where $r_m(d)$ is the rank of document $d$ in retriever $m$. This ensures documents appearing near the top of *both* dense and sparse searches get the highest final ranking.

---

### 11.2 Re-ranking Hybrid Search Strategies (`2-reranking (1).ipynb`) <a id="reranking" name="reranking"></a>

#### 🎯 Why We Need This:
Vector similarity search is fast, but it is **not always precise**. It evaluates candidate chunks independently against the query without understanding full cross-attention. 

**2-Stage Reranking Architecture**:
1. **Stage 1 (High Recall)**: Rapidly fetch a large candidate pool of chunks (e.g. `k=10` or `k=20`) from the Vector DB.
2. **Stage 2 (High Precision)**: Run a specialized **Cross-Encoder model** (e.g. `bge-reranker-base`) or LLM over the candidates to re-evaluate query-document pair context and return only the top 3 strictly relevant chunks.

#### Imports
```python
# Industry Standard Reranking using ContextualCompressionRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
```

#### How to Use (Standard 2-Stage Reranking Pipeline)

##### Method A: Cross-Encoder Reranker (Industry Standard & Fastest)
```python
# Stage 1: Base Retriever (High Recall - fetch top 10 candidates)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Stage 2: Cross-Encoder Reranker Model (High Precision - re-rank and pick top 3)
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=3)

# Combine into 2-Stage Compression Retriever
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
)

# Fetch top-3 re-ranked documents with high accuracy
reranked_docs = rerank_retriever.invoke("How can I use LangChain with memory?")
```

##### Method B: Simple LLM-based Reranker (No Extra Local Models Needed)
```python
# 1. Fetch initial candidate chunks
candidates = base_retriever.invoke(query)

# 2. Ask LLM to pick the top 3 most relevant documents directly
rerank_prompt = ChatPromptTemplate.from_template("""
Given the user question: "{question}"
Select the top 3 most relevant text chunks from the candidates below.

Candidates:
{candidates}

Return ONLY the text of the top 3 relevant chunks separated by '---'.
""")

rerank_chain = rerank_prompt | llm | StrOutputParser()
formatted_candidates = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(candidates)])
top_context = rerank_chain.invoke({"question": query, "candidates": formatted_candidates})
```

#### What They Do
*   `ContextualCompressionRetriever`: LangChain wrapper that takes a base retriever (Stage 1 high-recall search) and wraps it with a compressor/reranker (Stage 2 high-precision filter).
*   `CrossEncoderReranker` / `BAAI/bge-reranker-base`: Evaluates the query and candidate chunk jointly to assign a true relevance score, re-ordering chunks and keeping only `top_n=3`.
*   **Why 2-Stage Retrieval?**: Standard vector search can miss subtle nuances or return false positives. Reranking ensures only the single highest-quality context reaches the LLM context window.

---

### 11.3 Maximal Marginal Relevance - MMR (`3-mmr.ipynb`) <a id="mmr" name="mmr"></a>

#### 🎯 Why We Need This:
When a vector store contains multiple chunks from the same document or topic, standard top-k similarity search often returns **3 nearly identical chunks**. 
- Wasteful: Wastes context window tokens on redundant information.
- Incomplete: Misses other relevant, complementary perspectives across the corpus.

**MMR Solution**: MMR balances **Relevance to Query** against **Diversity among Retrieved Chunks** to maximize unique information density.

#### Imports
```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
```

#### How to Use
```python
# Create MMR Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5}
)

# Connect to RAG chain
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
response = rag_chain.invoke({"input": "How does LangChain support agents and memory?"})
```

#### What They Do
*   `MMR` (Maximal Marginal Relevance): Optimizes for both **similarity to query** and **diversity among selected documents**.
*   `fetch_k`: Number of candidate documents fetched initially for similarity (`20`).
*   `lambda_mult`: Controls trade-off between relevance and diversity (`1.0` = maximum relevance, `0.0` = maximum diversity, `0.5` = balanced).
*   **Key Concept**: MMR avoids retrieving multiple near-identical text chunks, maximizing topic coverage within the LLM context window.

<br>

---

<br>

### 11.4 RAG Search Strategies & Production Search Pipelines <a id="production-search" name="production-search"></a>

Retrieval performance directly governs the accuracy and grounding of a RAG pipeline. Below is a comprehensive breakdown of all major search paradigms used across modern AI systems.

#### 📊 Search Types in Retrieval-Augmented Generation

| Search Type | Description | Best For |
| :--- | :--- | :--- |
| **Keyword Search (Lexical Search)** | Matches exact terms using algorithmic statistical scoring like **BM25** or **TF-IDF**. | Exact domain terms, SKUs, product IDs, code snippets, error messages. |
| **Semantic Search (Vector Search)** | Uses dense vector embeddings to retrieve content based on underlying conceptual meaning rather than exact word matches. | Natural language queries, conversational questions, synonym/paraphrased matching. |
| **Hybrid Search** | Combines lexical (BM25) and dense vector search, merging score ranks via **Reciprocal Rank Fusion (RRF)**. | <mark style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px;">Industry standard</mark> for general RAG; delivers optimal precision + recall balance. |
| **Metadata Filtering** | Pre-filters or post-filters search spaces based on structured attributes (`author`, `date`, `category`, `tenant_id`). | Scoping queries to specific doc partitions, date ranges, or multi-tenant user access boundaries. |
| **Dense Retrieval** | Retrieves chunks by measuring high-dimensional vector distances (Cosine Similarity, Euclidean L2, Inner Product). | High-accuracy conceptual context matching. |
| **Sparse Retrieval** | Uses term-frequency weighted vectors (**BM25**, **SPLADE**) where most dimensions are zero. | Fast exact word matching with highly interpretable similarity scores. |
| **Multi-Vector Search** | Represents a single document using multiple embeddings (e.g., summary vector + full text vector + image vector). | Complex, multi-topic, or multi-modal documents. |
| **Hierarchical Search** | Performs two-tier retrieval: searches summary/coarse chunks first, then drills down into fine-grained sub-chunks. | Large enterprise manuals, books, lengthy technical specs. |
| **Parent-Child (Recursive) Retrieval** | Matches query against small, highly focused child chunks, but passes larger surrounding parent context to LLM. | Preserving deep context while keeping index chunking fine-grained. |
| **Multi-Query Retrieval** | Uses LLM to generate multiple prompt reformulations of user query, fetching vectors for all variants. | Resolving ambiguous, complex, or underspecified questions. |
| **Self-Query Retrieval** | Uses LLM to parse natural query into structured metadata query + semantic query payload. | Filtered requests (e.g., *"Find finance reports from 2024 covering AI"*). |
| **Graph-Based Retrieval (GraphRAG)** | Navigates entity nodes and relationship edges in a Knowledge Graph alongside vector embeddings. | Highly connected enterprise data, entity tracking, structural relationships. |
| **Reranked Search** | Fetches an expanded candidate pool (`k=20`), then re-scores candidate chunks using Cross-Encoders or Cohere Rerank. | Elevating top-3 context accuracy and removing false positives. |
| **Agentic Retrieval** | Autonomous agent decides dynamically which tool, strategy, or vector collection to query over multiple steps. | Multi-hop reasoning, live database queries, enterprise agent tool execution. |

<br>

#### 🏗️ Common Search Pipeline in Production RAG

```
┌───────────────────────────────────────────────┐
│                  User Query                   │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│     Metadata Filter (Pre-filtering Scope)     │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                 Hybrid Search                 │
│  ┌────────────────────┬────────────────────┐  │
│  │   BM25 (Sparse)    │  Vector (Dense)    │  │
│  └─────────┬──────────┴─────────┬──────────┘  │
└────────────┼────────────────────┼─────────────┘
             │                    │
             └──────────┬─────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│     Merge Results (Reciprocal Rank Fusion)    │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│      Reranker (Cross-Encoder / Cohere)        │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│        Top-K Context (High Precision)         │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│               LLM Response Output             │
└───────────────────────────────────────────────┘
```

<br>

#### 💡 Core Takeaway for Production Systems:
The most effective and widely adopted stack in enterprise production RAG relies on:
1. **Hybrid Search** (Dense Vector + Sparse BM25) for high retrieval recall.
2. **Metadata Pre-Filtering** to narrow security boundaries and date ranges.
3. **Cross-Encoder Reranking** to ensure top-3 chunks are strictly relevant to the prompt context.

<br>

---

<br>

## 12. Query Enhancement & Advanced RAG (`06_query_enhancment`) <a id="query-enhancement" name="query-enhancement"></a>

### 🧠 Section Overview: Why Do We Need Query Enhancement?
User questions in real-world applications are often **short**, **ambiguous**, **misspelled**, or **multi-part**.
- Direct vector search over raw user queries often fails because of **vocabulary mismatch** (users use colloquial words, while documents use formal/technical terminology).
- Multi-part queries (e.g. *"Compare product X and Y"*) fail because a single vector search cannot retrieve information about two distinct topics simultaneously.
- Short question sentences vector-align poorly with long descriptive paragraph chunks.

**Query Enhancement** uses LLM reasoning *before* searching the vector database to rewrite, expand, decompose, or generate hypothetical answers for optimal retrieval accuracy.

---

### 12.1 Query Expansion (`1-queryexpansion.ipynb`) <a id="query-expansion" name="query-expansion"></a>

#### 🎯 Why We Need This:
Users frequently submit short or vague queries (e.g., *"agent orchestration"*). 
- If the knowledge base uses different vocabulary (e.g., *"multi-agent coordination and DAG workflows"*), standard vector search fails to retrieve relevant chunks due to **vocabulary mismatch**.
- **Query Expansion** uses an LLM to automatically generate synonyms, related technical terms, and domain phrasing to broaden retrieval coverage.

#### Imports
```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
```

#### How to Use
```python
# Expansion prompt template
expansion_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Expand the following query to improve document retrieval by adding relevant synonyms, technical terms, and useful context.

Original query: "{query}"

Expanded query:
""")

expansion_chain = expansion_prompt | llm | StrOutputParser()
expanded_query = expansion_chain.invoke({"query": "What is agent orchestration?"})

# Retrieve documents using expanded query
retrieved_docs = retriever.invoke(expanded_query)
```

#### What They Do
*   `Query Expansion`: Uses an LLM pass to reformulate or enrich short, vague user prompts with technical vocabulary, domain synonyms, and context before retrieval.
*   **Key Concept**: Solves vocabulary mismatch issues where users ask questions using different vocabulary than what is written in the knowledge base.

---

### 12.2 Query Decomposition (`2-querydecomposition.ipynb`) <a id="query-decomposition" name="query-decomposition"></a>

#### 🎯 Why We Need This:
Complex, multi-part, or comparative questions (e.g., *"How does LangChain memory compare to CrewAI?"*) contain multiple distinct sub-topics.
- A single vector embedding for the entire prompt gets "averaged out" and fails to retrieve specific chunks for both subjects.
- **Query Decomposition** breaks down a complex question into sub-queries, executes sub-retrievals in parallel/sequence, and synthesizes a final composite answer.

#### Imports
```python
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableSequence
```

#### How to Use
```python
# 1. Prompt to decompose complex questions into sub-questions
decomposition_prompt = PromptTemplate.from_template("""
Decompose the following complex question into 2 to 4 smaller sub-questions for better document retrieval.

Question: "{question}"

Sub-questions:
""")
decomposition_chain = decomposition_prompt | llm | StrOutputParser()

# 2. Decompose and run sub-retrievals
sub_qs_text = decomposition_chain.invoke({"question": "How does LangChain memory compare to CrewAI?"})
sub_questions = [q.strip("-•1234567890. ").strip() for q in sub_qs_text.split("\n") if q.strip()]

# 3. Retrieve and answer each sub-question independently
results = []
for subq in sub_questions:
    docs = retriever.invoke(subq)
    # Combine retrieved docs for sub-question context
    context = "\n".join(d.page_content for d in docs)
    # Generate sub-answer
    sub_answer = qa_chain.invoke({"question": subq, "context": context})
    results.append(f"Sub-Q: {subq}\nSub-A: {sub_answer}")

# 4. Final Answer Synthesis (Combine sub-answers into comprehensive output)
final_prompt = PromptTemplate.from_template("""Combine the sub-answers to answer the original question: "{question}"\n\nContext:\n{sub_answers}""")
final_chain = final_prompt | llm | StrOutputParser()
final_response = final_chain.invoke({"question": "How does LangChain memory compare to CrewAI?", "sub_answers": "\n\n".join(results)})
```

#### What They Do
*   `Query Decomposition`: Breaks down multi-part or comparative questions into atomic sub-queries that can be retrieved independently.
*   **Key Concept**: Essential for multi-hop RAG pipelines and parallel agentic execution where a single query spans multiple distinct domain areas.

---

### 12.3 Hypothetical Document Embeddings - HyDE (`3-HyDE.ipynb`) <a id="hyde" name="hyde"></a>

#### 🎯 Why We Need This:
Standard vector search compares a **short question vector** (e.g. *"When was NeXT founded?"*) against **long answer paragraph vectors**. 
- Because questions and answers look structurally different, vector similarity can be weak (**query-document asymmetry**).
- **HyDE Solution**: HyDE asks the LLM to write a *hypothetical answer passage* first. It then embeds that generated answer and uses it to search the Vector DB for real documents that look like the hypothetical passage.

#### Imports
```python
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

#### How to Use
```python
# 1. Generate hypothetical document/answer using LLM
hyde_prompt = PromptTemplate.from_template("""
Please write a passage to answer the question.

Question: "{question}"
Passage:
""")
hyde_chain = hyde_prompt | llm | StrOutputParser()
hypothetical_doc = hyde_chain.invoke({"question": "When did Steve Jobs found NeXT?"})

# 2. Embed hypothetical document and search vector store
retrieved_docs = vectorstore.similarity_search(hypothetical_doc, k=3)
```

#### What They Do
*   `HyDE`: Generates a hypothetical response document using an LLM, embeds that hypothetical passage, and uses its vector to search the vector database.
*   **Key Concept**: Eliminates query-document asymmetry. Standard queries are short question sentences, whereas stored chunks are detailed descriptive paragraphs. Vector matching an answer-like text against stored documents yields significantly higher similarity alignment.

---

## 13. Multimodal RAG (`07_multimodle RAG`) <a id="multimodal-rag" name="multimodal-rag"></a>

> [!NOTE]
> Section 13 (Multimodal RAG) and subsequent topics have been transferred and expanded in **[notes2.md](file:///c:/Users/DELL/Desktop/rag_praacties/notes2.md)**.
> Please refer to **[notes2.md](file:///c:/Users/DELL/Desktop/rag_praacties/notes2.md)** for detailed documentation on `1-multimodalopenai.ipynb` and future RAG modules.

