
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader

def main():
    load_dotenv()
    st.set_page_config(page_title='Document Reader', layout="wide")
    st.header('🌈 Know your Vermont 💭')
    st.subheader(':green[Search from diretory or upload a file]')
    
    col1, col2 = st.columns(2, gap="large")
    col1.button("Directory", disabled=True)
    col2.button("Upload", disabled=True)

    # local directory
    col1.write('📂 Data will be feeded from local directory')
    loader = DirectoryLoader('documents', glob='**/*.pdf')
    docs = loader.load()
    char_text_splitter = CharacterTextSplitter(separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
    )
    
    doc_texts = char_text_splitter.split_documents(docs)
    openAI_embeddings = OpenAIEmbeddings()
    
    vStore = Chroma.from_documents(doc_texts, openAI_embeddings)
    model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vStore)
    user_question = col1.text_input("Ask a question from directory data !")
    if user_question:
        response = model.run(user_question)
        col1.write(response)


    # upload file
    pdf = col2.file_uploader("📌 Upload your pdf doc to feed data", type="pdf")
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = col2.text_input("Ask a question from file data !")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            col2.write(response)



if __name__ == '__main__':
    main()
