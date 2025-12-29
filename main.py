import os
import time
import pickle
import streamlit as st

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="AlphaLens", layout="wide")
st.title("🚀 AlphaLens — See the Market Beyond the News 📈")


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
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.9,
    max_tokens=500,
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

        retriever = vectorstore.as_retriever()

        # Get relevant documents for the query
        docs = retriever.get_relevant_documents(query)

        # Load a QA-with-sources chain and run it on the retrieved docs
        chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        result = chain({"input_documents": docs, "question": query})

        # Robustly extract answer text from known keys
        answer = result.get("answer") or result.get(
            "output_text") or result.get("result") or result.get("final_answer")

        st.header("📝 Answer")
        st.write(answer or "No answer found.")

        # Extract sources: try known keys, fall back to metadata on source_documents
        sources = result.get("sources")
        if not sources:
            src_docs = result.get("source_documents") or result.get(
                "source_documents")
            if src_docs:
                try:
                    sources = "\n".join(d.metadata.get(
                        "source", str(d)) for d in src_docs)
                except Exception:
                    sources = "\n".join(str(d) for d in src_docs)

        if sources:
            st.subheader("🔗 Sources")
            for source in sources.split("\n"):
                st.write(source)

    except Exception as e:
        st.error(f"❌ Error while answering question: {e}")
