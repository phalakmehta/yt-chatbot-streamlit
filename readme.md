**▶️ YT Chatbot**

*RAG-powered YouTube Video Chat · LangChain · FAISS · Groq*

![](media/49503effaa774dc9577a11a8f1007c6b8f03b024.png){width="5.208333333333333in" height="3.2291666666666665in"}

What is it?

YT Chatbot lets you paste any YouTube URL and chat with its content. It fetches the video transcript, indexes it using vector embeddings, and answers your questions using an LLM --- all without watching the video.

How RAG is Implemented

RAG (Retrieval-Augmented Generation) is the core of this app. Here\'s how it works step by step:

**1. Transcript Extraction**

> The YouTube transcript is fetched using **youtube-transcript-api**. This gives us the raw text of everything spoken in the video.

**2. Chunking**

> The transcript is split into overlapping chunks of 1000 characters (200 overlap) using LangChain\'s **RecursiveCharacterTextSplitter**. Overlap ensures context isn\'t lost at chunk boundaries.

**3. Embedding & Indexing**

> Each chunk is converted into a vector embedding using **HuggingFace\'s all-MiniLM-L6-v2** model (runs locally, no API needed). These vectors are stored in a **FAISS** in-memory vector store for fast similarity search.

**4. Retrieval**

> When a user asks a question, it is also embedded and the top 4 most semantically similar chunks are retrieved from FAISS using cosine similarity.

**5. Generation**

> The retrieved chunks + the user\'s question are passed to **Groq\'s LLaMA3-8b** via LangChain\'s LCEL pipeline. The LLM answers strictly from the provided context.

**YouTube URL → Transcript → Chunks → FAISS Index → Retriever → LLM → Answer**

Tech Stack

**• Frontend:** Streamlit

**• Embeddings:** HuggingFace all-MiniLM-L6-v2 (local)

**• Vector Store:** FAISS (in-memory)

**• LLM:** Groq API (openai/gpt-oss-120b)

**• Orchestration:** LangChain LCEL

**• Deployment:** Streamlit Cloud

Live Demo

**https://yt-chatbot-app-ujefappxae2z73fnr5tkibr.streamlit.app**
