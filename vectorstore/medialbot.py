import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())
DB_FAISS_PATH = "E:\\Medical Chatbot\\vectorstore\\db_faiss"

# Cache the loaded vectorstore for performance
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template: str):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id: str, hf_token: str):
    if not hf_token:
        st.error("HuggingFace API token not found. Please set HF_TOKEN in your environment.")
        return None
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=hf_token,
        temperature=0.5,
        task="text-generation",
        max_new_tokens=512
    )
    return llm

def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
Don't provide anything outside of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

        HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"  # Replace with your model repo if needed
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return

            llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN)
            if llm is None:
                return  # Error shown inside load_llm

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            # Check if any relevant source document was returned
            if not source_documents or result.strip().lower() in ["", "i don't know", "i do not know"]:
                result_to_show = "🔍 **Answer not found in the database. Please ask a medical-related question.**"
            else:
                result_to_show = result + "\n\n**Source Documents:**\n"
                for i, doc in enumerate(source_documents):
                    result_to_show += f"{i+1}. {doc.page_content}\n"

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
