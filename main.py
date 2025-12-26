import os
import time
import pickle
import streamlit as st

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="RockyBot", layout="wide")
st.title("🧠 RockyBot: News Research Tool 📈")

st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

# -------------------- LLM --------------------
llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",  # replacement for text-davinci-003
    temperature=0.9,
    max_tokens=500
)

# -------------------- Process URLs --------------------
if process_url_clicked and urls:
    try:
        # Load articles
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("🔄 Loading articles...")
        data = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(data)
        main_placeholder.text("✂️ Splitting text into chunks...")

        # Embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )

        # Vector store
        vectorstore = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("📦 Building FAISS index...")
        time.sleep(1)

        # Save FAISS index
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("✅ URLs processed successfully! You can now ask questions.")

    except Exception as e:
        st.error(f"❌ Error while processing URLs: {e}")

# -------------------- Query Section --------------------
query = st.text_input("Ask a question about the articles:")

if query and os.path.exists(file_path):
    try:
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # IMPORTANT FIX: chain_type="stuff"
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff"
        )

        result = chain({"question": query}, return_only_outputs=True)

        st.header("📝 Answer")
        st.write(result.get("answer", "No answer found."))

        sources = result.get("sources", "")
        if sources:
            st.subheader("🔗 Sources")
            for source in sources.split("\n"):
                st.write(source)

    except Exception as e:
        st.error(f"❌ Error while answering question: {e}")
