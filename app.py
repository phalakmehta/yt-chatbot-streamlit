import os
import re
import streamlit as st

st.set_page_config(page_title="YT Chatbot", page_icon="▶️", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0d0d0f; color: #e8e8f0; }
    .stTextInput input { background-color: #141417; color: #e8e8f0; border: 1px solid #222228; }
    .stButton button { background: linear-gradient(135deg, #ff3f3f, #ff7a3f); color: white; border: none; border-radius: 8px; }
    .user-msg { background: #1a1a24; padding: 10px 14px; border-radius: 10px; margin: 6px 0; border: 1px solid #2e2e38; }
    .bot-msg { background: #111116; padding: 10px 14px; border-radius: 10px; margin: 6px 0; border: 1px solid #222228; }
    .label { font-size: 11px; font-weight: 500; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("▶️ YT Chatbot")
st.caption("RAG-powered video chat · LangChain · FAISS · Groq")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    if re.match(r"^[A-Za-z0-9_-]{11}$", url):
        return url
    raise ValueError("Could not extract video ID from URL")


@st.cache_resource
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=os.environ.get("GROQ_API_KEY"),
    )


def build_chain(video_id: str):
    from youtube_transcript_api import YouTubeTranscriptApi
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"]).to_raw_data()
    transcript = " ".join(chunk["text"] for chunk in transcript_list)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(chunks, get_embeddings())
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
        """,
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })
        | prompt | get_llm() | StrOutputParser()
    )
    return chain


with st.container():
    url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    if st.button("Load Video", use_container_width=True):
        if url.strip():
            with st.spinner("Fetching transcript & building index..."):
                try:
                    vid = extract_video_id(url.strip())
                    chain = build_chain(vid)
                    st.session_state.chain = chain
                    st.session_state.video_id = vid
                    st.session_state.messages = []
                    st.success(f"✅ Ready! Video ID: `{vid}`")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a YouTube URL.")

st.divider()

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="label" style="color:#ff7a3f90">You</div><div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="label" style="color:#ff3f3f80">Assistant</div><div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

if st.session_state.chain:
    question = st.chat_input("Ask about the video...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.chain.invoke(question)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
        st.rerun()
else:
    st.info("⬆️ Load a YouTube video above to start chatting.")