import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tiktoken

# Loading the OpenAI API key from .env
load_dotenv(find_dotenv(), override=True)
api_key = os.environ.get('OPENAI_API_KEY')

# Clear the chat history from Streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

# Load a document based on its extension
def load_document(file):
    _, extension = os.path.splitext(file)

    if extension == '.pdf':
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        loader = TextLoader(file)
    else:
        st.error('Document format is not supported!')
        return None

    data = loader.load()
    return data

# Split data into chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

# Ask a question and get an answer using LangChain's RetrievalQA
def ask_and_get_answer(vector_store, q, k=3):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer

# Calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004

# Streamlit UI
if __name__ == "__main__":
    st.title("ask-docs")

    # Define your navigation items
    # nav_items = ["Home", "About", "Services", "Contact"]
    st.image('files/img.png')
    st.subheader('''LLM QA Application''')

    with st.sidebar:
        # Text input for the OpenAI API key:: part of the plan to be...till havnt added
        # api_key = st.text_input('OpenAI API Key:', type='password', value=api_key)
        os.environ['OPENAI_API_KEY'] = api_key

        # File uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # Chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # Add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking, and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked, and embedded successfully.')

    # User's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)

            # Text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer)

            st.divider()

            if 'history' not in st.session_state:
                st.session_state.history = ''

            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # Text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)
