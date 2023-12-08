import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from HtmlCss import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer
from langchain.llms import Replicate
from langchain.llms import CTransformers
from transformers import AutoModelForCausalLM



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    llm = CTransformers( model ="models\mistral-7b-instruct-v0.1.Q8_0.gguf",
    temperature=0.05,
    top_p=1, 
    verbose=True,
    n_ctx=4096,
    context_length= 6000,
    tokenizer=tokenizer,  # Pass tokenizer explicitly
)
    # llm = CTransformers(
    #                                        model = "models\llama-2-7b-chat.ggmlv3.q8_0.bin", 
    #                                        model_type="llama", 
    #                                        gpu_layers=50,
    #                                        max_new_tokens = 1000,
    #                                        temperature= 0.05,
    #                                        context_length = 6000)
    
#     llm = CTransformers( model = "models\llama-2-7b-chat.ggmlv3.q8_0.bin",
                        
#                         model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01, 'context_length': 1024,
#         },
#     tokenizer=tokenizer,  
# )
    # llm = Replicate(
    #     # streaming = True,
    #     model = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
    #     # callbacks=[StreamingStdOutCallbackHandler()],
    #     model_kwargs = {"temperature": 0.01, "max_length" :500,"top_p":1})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type='pdf', accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()