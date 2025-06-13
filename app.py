import gradio as gr
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  

def handle_file(text_input, file):
    if file is None:
        return "Nenhum arquivo enviado."
    
    # 1. Carrega o documento
    loader = PyPDFLoader(file)
    docs = loader.load()

    # 2. Divide em partes menores (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # 3. Gera embeddings com o modelo local
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # 4. Cria o índice vetorial (InMemoryVectorStore)
    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

    # 5. Cria o LLM e a cadeia RAG
    llm = ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=os.environ.get("OPENAI_ORGANIZATION_KEY"),
        model="gpt-4o-mini",
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # 6. Faz uma pergunta
    query = text_input
    response = qa_chain.invoke({"query": query})

    return response["result"]

    
with gr.Blocks() as demo:
    gr.Markdown("## Verificador de PDFs com RAG")
    text_input = gr.Textbox(label="Entre com sua questão aqui", placeholder="Faça uma pergunta sobre o PDF carregado")
    upload = gr.File(type="filepath")
    output = gr.Textbox(label="Output")
    button = gr.Button("Submit")
    
    button.click(
        fn=handle_file,
        inputs=[text_input, upload],
        outputs=output
    )
demo.launch(share=True)

