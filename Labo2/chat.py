from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import hub
from langchain.prompts import PromptTemplate

import chainlit as cl

custom_prompt = """
You are an assistant for question-answering tasks but you talk like Mario the plumber. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""


def process_pdf_file(pdf_file : cl.types.AskFileResponse)->list[Document]:
    loader = PyPDFLoader(pdf_file.path)  
    pages = loader.load()

    # print(pages)

    # Use RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=0)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)

    return docs

def create_vector_store(docs: list[Document]) -> Chroma:
    embeddings = GPT4AllEmbeddings()
    chroma = Chroma.from_documents(docs, embeddings)
    return chroma

def pdf_file_vector_store(pdf_file):
    # Transform pdf file into vector/embeddings
    return create_vector_store(process_pdf_file(pdf_file))

def load_llm():
    model = Ollama(model="llama2")  
    return model

def retrieval_qa_chain(pdf_file):
    # TODO 5: Load prompt ("rlm/rag-prompt")

    # prompt = hub.pull("rlm/rag-prompt")
    prompt = PromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks but you talk like Mario the plumber. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise
        Question: {question} 
        Context: {context} 
        Answer:
        """
    )

    # TODO 6: Use RetrievalQA to create the prompt and load the model + embeddings.
    model = load_llm()

    docs = process_pdf_file(pdf_file)
    chroma = create_vector_store(docs)

    qa_chain = RetrievalQA.from_chain_type(
        model, retriever=chroma.as_retriever(), chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


@cl.on_chat_start
async def start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!", accept=["application/pdf"], max_files=1, max_size_mb=100,
        ).send()

    pdf_file = files[0]

    print(pdf_file)

    # Notify the user that we are loading stuff.
    message = cl.Message(
        content=f"Processing `{pdf_file.name}` file and loading model.."
    )
    await message.send()    

    qa_chain = retrieval_qa_chain(pdf_file)

    # Notify the user that everything is loaded.
    message.content = f"`{pdf_file.name}` file has been processed. Feel free to ask any questions about it !"
    await message.update()

    print("QA Chain loaded successfully !")
    # print(qa_chain)
    # Saves the qa_chain into the user session.
    cl.user_session.set("qa_chain", qa_chain)
    #TODO 8

@cl.on_message
async def main(message):
    qa_chain=cl.user_session.get("qa_chain")

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached=True

    # Use the chain for the question
    res = await qa_chain.ainvoke(message.content, callbacks=[cb])

    # Retrieve the answer result
    answer=res["result"]

    # Send the answer to the chat
    await cl.Message(content=answer).send() 
