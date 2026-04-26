#  Personal Research Assistant

Writing a thesis is one of the most demanding parts of a university degree, especially in the final year. Students are often required to go through 10, 20, or sometimes even 50 research papers just to build a solid literature review and define a clear project methodology. This process is not only time-consuming but also mentally intensive, requiring sustained focus to extract and compare relevant insights across dense academic material.

Having gone through this experience myself, I realized how challenging it is to efficiently synthesize information across multiple papers while keeping track of key ideas such as methodologies, datasets, and model choices. A significant amount of time is spent repeatedly reading, summarizing, and cross-referencing information that could otherwise be streamlined.

This project was built to address that problem.

So the idea is to create a RAG-powered assistant that allows users to upload research papers and interact with them conversationally. Instead of manually scanning through documents, users can ask targeted questions such as:
- Comparing methodologies across papers  
- Identifying pros and cons of different approaches  
- Understanding datasets used in each study  
- Summarizing key contributions across multiple sources  

Rather than simply summarizing individual documents, the system is designed to **connect information across papers**, enabling comparative analysis and faster knowledge synthesis.

In essence, this project combines a personal pain point with modern AI techniques — RAG, vector embeddings, and LLMs — to create a more efficient and intuitive way of engaging with academic literature.

---

## Features Implemented and Concepts Showcased

The goal of this project is to build a **personal research assistant** that can:

- Ingest multiple research papers (pdf documents)
- Extract and chunk textual content
- Generate semantic embeddings for each chunk
- Store and retrieve chunks using similarity search (FAISS)
- Answer user questions using retrieved context (RAG pipeline)

This project demonstrates core concepts in:
- Information Retrieval
- Embedding-based search
- Vector databases (FAISS)
- Large Language Model (LLM) integration (Gemini Integration)
- End-to-end RAG architecture

---

##  System Architecture

### 1. PDF Processing
- PDFs are parsed and converted into raw text
- Text is split into smaller overlapping chunks

### 2. Embedding Generation
- Each chunk is converted into a dense vector using **Gemini Embeddings API**
- Queries are also embedded in the same vector space

### 3. Vector Store (FAISS)
- Embeddings are stored in a FAISS index
- Enables fast similarity search over large document sets

### 4. Retrieval
- User query is embedded
- Top-K most similar chunks are retrieved using L2 distance

### 5. Answer Generation (RAG)
- Retrieved chunks are passed to a generative model
- Model generates a final contextual answer based on provided context

---

##  Tech Stack

- **Frontend:** Streamlit  
- **Backend Logic:** Python  
- **Embeddings:** Google Gemini Embedding API  
- **LLM:** Gemini Flash 
- **Vector Database:** FAISS (Facebook AI Similarity Search)  
- **PDF Parsing:** PyPDF-based utility  
- **Data Handling:** NumPy, Pickle  

---

##  Project Structure

├── app.py    
├── utils/    
│ ├── pdf_parser.py    
│ ├── chunking.py    
│ ├── embeddings.py    
│ ├── retrieval.py    
│ └── rag_pipeline.py    
├── cache/    
│ └── (stored FAISS indexes per document)    
├── requirements.txt    
└── README.md    


A glimpse of how the project works: [Personal Research Assistant Video.webm](https://github.com/user-attachments/assets/28cbc9f8-e08b-4857-ae4b-808ea9f4e86d)


### Instructions to run the Project:

1) Install dependencies
```{bash}
pip install -r requirements.txt
```

2) Add an API key
Create a .env file under the research copilot folder (main project folder) and add the following line:
```{txt}
GEMINI_API_KEY=paste_your_key_here
```

3) Run the app
```{bash}
streamlit run app.py
```
