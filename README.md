# StudyMate

A conversational RAG study assistant that answers questions from the Class 7 NCERT Science textbook.

StudyMate combines a Streamlit chat interface with history-aware retrieval, Pinecone vector search, local BGE embeddings, and OpenAI chat models. It returns concise, textbook-grounded answers and suggests related questions to continue the learning session.

## Features

- Answers questions against 13 bundled NCERT Science chapter PDFs
- Rewrites follow-up questions into standalone queries using conversation history
- Retrieves six diverse passages from Pinecone with maximal marginal relevance
- Uses hierarchical 2,000-character parent and 400-character child chunks during ingestion
- Runs BGE embeddings on CUDA when available and falls back to CPU
- Generates two contextual follow-up questions after each answer
- Limits each browser session to five student questions

## Architecture

```mermaid
flowchart LR
    books[NCERT Science PDFs] --> indexing[Load, split, and embed]
    indexing --> pinecone[(Pinecone index)]

    student[Student question] --> ui[Streamlit chat]
    ui --> workflow[History-aware MMR retrieval<br/>and context prompt]
    pinecone --> workflow
    workflow --> models[OpenAI answer and<br/>follow-up generation]
    models --> response[Answer and suggestions]

    classDef entry fill:#0969DA,stroke:#79C0FF,color:#FFFFFF,stroke-width:2px
    classDef process fill:#334155,stroke:#CBD5E1,color:#FFFFFF,stroke-width:2px
    classDef action fill:#6D28D9,stroke:#C4B5FD,color:#FFFFFF,stroke-width:2px
    classDef service fill:#166534,stroke:#86EFAC,color:#FFFFFF,stroke-width:2px
    classDef output fill:#9F1239,stroke:#FDA4AF,color:#FFFFFF,stroke-width:2px

    class books,student entry
    class indexing,workflow process
    class pinecone,models service
    class ui,response output
```

## Tech stack

- Python and Streamlit
- LangChain conversational retrieval chains
- OpenAI `gpt-3.5-turbo-0125` for answers and `gpt-3.5-turbo-1106` for follow-up questions
- Pinecone serverless vector storage
- `BAAI/bge-small-en` sentence-transformer embeddings
- PyPDF for textbook ingestion

## Getting started

### Prerequisites

- Python 3.10 or newer
- An OpenAI API key
- A Pinecone API key and an unused index name

### 1. Install the project

```bash
git clone https://github.com/that-rookie-parth/StudyMate.git
cd StudyMate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

### 2. Configure credentials

```bash
cp .env.example .env
```

Add your values to `.env`:

```dotenv
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="your-index-name"
OPENAI_API_KEY="your-openai-api-key"
```

The populated `.env` file is ignored by Git. Legacy `pinecone_index_name` and `key` variables remain supported for existing local configurations.

### 3. Index the textbook

> [!WARNING]
> `upload_data.py` permanently deletes any existing Pinecone index whose name matches `PINECONE_INDEX_NAME`, then recreates it in AWS `us-east-1`. Use a new, project-specific index name.

```bash
python upload_data.py
```

The script loads the PDFs from `Content/`, creates hierarchical chunks, embeds them into 384-dimensional vectors, and uploads them to the `studymate_chatbot` namespace.

### 4. Run the app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit and ask a question from the textbook, such as “How do plants prepare their food?”

## Project structure

```text
StudyMate/
├── Content/          # 13 NCERT Science chapter PDFs
├── app.py            # Streamlit interface and session limits
├── chat_logic.py     # Conversational RAG chain and chat history
├── db.py             # Embeddings, Pinecone connection, and retriever
├── upload_data.py    # Destructive index creation and document ingestion
├── utils.py          # Structured follow-up-question generation
└── requirements.txt  # Pinned Python dependencies
```

## Limitations

- This is an educational prototype, not a production learning platform.
- The configured GPT-3.5 model snapshots are legacy models and may need replacement as provider availability changes.
- Chat history is kept only in Streamlit session memory and is not persisted.
- Answers do not expose retrieved passages or textbook citations in the interface.
- The app has no authentication, moderation layer, automated tests, or deployment configuration.
- The ingestion script replaces the configured Pinecone index instead of updating it incrementally.

## License and textbook content

The source code is available under the [MIT License](LICENSE). The bundled NCERT textbook PDFs are third-party educational material and are not covered by the software license; their original rights and terms remain with their publisher.
