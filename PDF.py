import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import faiss
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from HtmlCss import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

# a=input("")

def get_docs_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 100,
        chunk_overlap=20,
        length_function= len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):

    embeddings = HuggingFaceInstructEmbeddings(model_name ="hkunlp/instructor-xl")
    vectorstor = faiss.FAISS.from_texts(texts= text_chunks, embedding=embeddings)
    return vectorstor

def get_ConversationChain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.9, "max_length":1500})
    memory = ConversationBufferMemory(memory_key= 'Chat_history', return_messages= True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF analyzer", page_icon= ":books:" )
    st.write(css, unsafe_allow_html=True)

    if "Conversation" not in st.session_state:
        st.session_state.ConversationChain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDF analyzer :books:")
    user_question = st.text_input("Please ask all your questions here:")
    if user_question:
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Documents here")
        docs = st.file_uploader("Upload files here", type="PDF", accept_multiple_files=True)
        if st.button("Analyze"):
            with st.spinner("In-Progress"):
                raw_text = get_docs_text(docs)
                
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.Conversation = get_ConversationChain(vectorstore)


if __name__== '__main__':
    main()
    