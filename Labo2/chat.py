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

import chainlit as cl

def process_pdf_file(pdf_file : cl.types.AskFileResponse)->list[Document]:
    # Load documents with PyPDFLoader
    loader = #TODO 1
    document = #TODO 1

    # Use RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=0)
    text_splitter = #TODO 2
    docs = #TODO 2

    return docs

def create_vector_store(docs: list[Document]) -> Chroma:
    # Transform the documents into embeddings using GPT4AllEmbeddings.
    return #TODO 3

def pdf_file_vector_store(pdf_file):
    # Transform pdf file into vector/embeddings
    return create_vector_store(process_pdf_file(pdf_file))

def load_llm():
    return #TODO 4

def retrieval_qa_chain(pdf_file):
    # Load prompt ("rlm/rag-prompt")
    prompt = #TODO 5

    # Use RetrievalQA to create the prompt and load the model + embeddings.
    qa_chain = #TODO 6
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

    # Notify the user that we are loading stuff.
    message = cl.Message(
        content=f"Processing `{pdf_file.name}` file and loading model.."
    )
    await message.send()
    
    qa_chain = #TODO 7

    # Notify the user that everything is loaded.
    message.content = f"`{pdf_file.name}` file has been processed. Feel free to ask any questions about it !"
    await message.update()

    # Saves the qa_chain into the user session.
    #TODO 8

@cl.on_message
async def main(message):
    # Retrieve the qa_chain saved in the session
    qa_chain= #TODO 9

    cb = cl.AsyncLangchainCallbackHandler(
    stream_final_answer=True,
    answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached=True

    # Use the chain for the question
    res = await qa_chain.acall(message.content, callbacks=[cb])

    # Retrieve the answer result
    answer=res["result"]

    # Send the answer to the chat
    await cl.Message(content=answer).send() 
