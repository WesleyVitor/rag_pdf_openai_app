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

class HandleLLMUseCase:
    
    def __init__(self, text_input, file):
        self.text_input = text_input
        self.file = file
        self.docs = None
        self.chunks = None
        self.retriever = None
        self.qa_chain = None

    def handle_pdf_load(self):
        """
        Carrega o PDF enviado e extrai os documentos.
        Se nenhum arquivo for enviado, levanta um erro.
        """
        if self.file is None:
            raise ValueError("Nenhum arquivo enviado.")
        
        loader = PyPDFLoader(self.file)
        self.docs = loader.load()

    def handle_text_split(self):
        """
        Divide os documentos carregados em chunks menores de 
        500 caracteres com sobreposição de 50 caracteres.
        Se nenhum documento for carregado, levanta um erro.
        """
        if not hasattr(self, 'docs'):
            raise ValueError("Nenhum documento carregado.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.chunks = text_splitter.split_documents(self.docs)

    def handle_embeddings(self):
        """
        Cria embeddings dos chunks de texto usando OpenAIEmbeddings .
        Se nenhum chunk de texto estiver disponível, levanta um erro.
        """
        if not hasattr(self, 'chunks'):
            raise ValueError("Nenhum chunk de texto disponível.")
        
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        vectorstore = InMemoryVectorStore.from_documents(self.chunks, embeddings)
        self.retriever = vectorstore.as_retriever()
    
    def handle_chain_creation(self):
        """
        Cria uma cadeia de perguntas e respostas (RetrievalQA) usando o modelo LLM.
        """
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORGANIZATION_KEY"),
            model="gpt-4o-mini",
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.retriever
        )
    
    def handle_query(self):
        """
        Executa a consulta na cadeia de perguntas e respostas e retorna o resultado.
        """
        response = self.qa_chain.invoke({"query": self.text_input})
        return response["result"]

    def execute(self):
        self.handle_pdf_load()
        self.handle_text_split()
        self.handle_embeddings()
        self.handle_chain_creation()
        response = self.handle_query()
        return response



def handle_file(text_input, file):
    instance = HandleLLMUseCase(text_input, file)
    return instance.execute()

    
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

