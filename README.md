# Chat with Your Code – RAG for C/C++ Projects  

🚀 **Chat with Your Code** is an AI-powered system that allows developers to ask natural language questions about a C/C++ codebase and receive meaningful, context-aware explanations.  
It combines **Retrieval-Augmented Generation (RAG)** with **LLM-based reasoning** to make navigating large, complex codebases easier.  

---

## ✨ Features  

- 📂 **Codebase Ingestion** – Parses `.c`, `.cpp`, and `.h` files into semantically meaningful chunks (functions, comments, docstrings).  
- 🔍 **Semantic Search** – Uses embeddings + vector DB (ChromaDB) to retrieve relevant code snippets.  
- 🤖 **AI-Powered Q&A** – LLM answers questions strictly grounded in retrieved context.  
- 📝 **Detailed Explanations** – Provides line-by-line or logical flow breakdown of functions.  
- 🎨 **Mermaid Diagrams** – Automatically generates flowcharts for functions to visualize execution flow.  
- 💻 **Interactive Frontend** – Streamlit app with query box, retrieved code snippets, and rendered diagrams.  
- 🌳 **Semantic Chunking with AST** – Instead of naive text-splitting, leverages **tree-sitter** to parse C/C++ into an Abstract Syntax Tree (AST). Chunks are based on logical syntax units (functions, classes, structs, comments).  
- 🏷️ **Rich Metadata Extraction** – Each chunk is enriched with metadata such as `type`, `function_name`, `class_name`, `struct_name`, and `start_line`, stored in ChromaDB for precise retrieval.  
- ⚡ **Efficient Vector Retrieval** – Natural language queries are embedded via SentenceTransformer and matched against stored embeddings in ChromaDB for accurate similarity search.  
- 🔄 **Robust Ingestion Pipeline** – Handles new source files with batching, unique chunk IDs, metadata enrichment, and efficient embedding storage in ChromaDB.  


---

## 🏗️ System Architecture  

```mermaid
flowchart TD
    A[User Query] --> B[FastAPI Backend /ask Endpoint]
    B --> C[RAG Module: Retrieve Relevant Code Chunks]
    C --> D[LLM Module: Gemini 1.5 Flash / GPT]
    D --> E[Answer + Mermaid Diagram]
    E --> F[Streamlit Frontend]
    F --> G[Display: Text + Retrieved Code + Diagram]

---

## 🛠️ Tech Stack  

| Technology                     | Purpose                                                                 |
|--------------------------------|-------------------------------------------------------------------------|
| **Python**                     | Core programming language for backend logic                            |
| **FastAPI**                    | Backend framework exposing the `/ask` API endpoint                     |
| **React.js**                   | Frontend framework for building the interactive UI                     |
| **tree-sitter**                | Parser that generates AST for C/C++ to enable semantic chunking         |
| **tree-sitter-languages**      | Provides precompiled grammars for parsing C and C++ code                |
| **SentenceTransformer (all-MiniLM-L6-v2)** | Creates vector embeddings for code chunks & queries |
| **ChromaDB**                   | Lightweight vector database for storing embeddings + metadata           |
| **Mermaid.js**                 | Renders diagrams (flowcharts) in the frontend                          |
| **dotenv**                     | Manages environment variables (API keys, configs)                      |
| **Uvicorn**                    | ASGI server to run FastAPI backend                                     |

---



## 📂 Project Structure  

```bash
chat-with-code/
│── backend/                 
│   │── main.py              # FastAPI backend with /ask endpoint
│   │── rag_module.py        # Vector DB (ChromaDB) + C/C++ function & comment chunking logic + retrieval logic
│   │── llm_module.py        # Gemini/GPT integration with strict system prompt
│
│── frontend/                # react-based UI              
│── data/                    # Sample C/C++ codebases
│── requirements.txt         # Python dependencies
│── README.md                # Project documentation

---

##⚙️ Installation
'''bash
# 1. Clone the repo
git clone https://github.com/your-username/chat-with-code.git
cd chat-with-code

# 2. Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
'''

**Create a .env file in the root directory:**
'''bash
GEMINI_API_KEY=your_google_gemini_key
'''

**Run the backend:**
'''bash
uvicorn backend.main:app --reload
'''

**Run Frontend:**
'''bash
npm run dev
'''
---



