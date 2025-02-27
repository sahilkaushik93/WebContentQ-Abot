import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import os
import pickle
import concurrent.futures

st.title("Web Content Q&A Tool (Streamlit Hosted)")

VECTOR_STORE_PATH = "vector_store.pkl"

def model_inference(lst):
    i = 0
    while i < len(lst):
        try: 
            i+=1
            llm = HuggingFaceHub(
                repo_id= lst[i],
                model_kwargs={"temperature": 0.7},
                huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
            )
            return llm
        except Exception as e:
            i+=1

# Load cached vector store if available
if os.path.exists(VECTOR_STORE_PATH):
    with open(VECTOR_STORE_PATH, "rb") as f:
        vector_store = pickle.load(f)
    retriever = vector_store.as_retriever()
    st.session_state["retriever"] = retriever
    st.success("Loaded cached content.")

urls = st.text_area("Enter URLs (one per line):").split("\n")

if st.button("Fetch & Process Content"):
    if not urls or urls == [""]:
        st.warning("Please enter at least one URL.")
    else:
        with st.spinner("Fetching content..."):
            def fetch_content(url):
                try:
                    loader = WebBaseLoader(url)
                    return loader.load()
                except Exception as e:
                    st.error(f"Error loading {url}: {str(e)}")
                    return []

            # Fetch in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(fetch_content, urls)

            documents = [doc for result in results for doc in result]

        with st.spinner("Processing content..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)

            # Use Hugging Face embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever()

            # Cache processed data
            with open(VECTOR_STORE_PATH, "wb") as f:
                pickle.dump(vector_store, f)

            st.session_state["retriever"] = retriever
            st.success("Content processed successfully! You can now ask questions.")

question = st.text_input("Ask a question about the content:")

if st.button("Get Answer"):
    if "retriever" not in st.session_state:
        st.warning("Please fetch content first.")
    elif not question:
        st.warning("Enter a question to get an answer.")
    else:
        with st.spinner("Searching for the answer..."):
            # Hugging Face API-based LLM
            huggingfacemodels = st.secrets["HUGGINGFACEHUB_MODELS"]

            # llm = HuggingFaceHub(
            #     repo_id="google/gemma-2b",
            #     model_kwargs={"temperature": 0.7},
            #     huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
            # )

            llm = model_inference(huggingfacemodels)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state["retriever"]
            )
            answer = qa_chain.run(question)
            st.success(answer)
