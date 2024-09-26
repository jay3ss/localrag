import pathlib

from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

EMBEDDING_MODEL = "nomic-embed-text"
INFERENCE_MODEL = "llama3.1"

docs_dir = (
    pathlib.Path.home()
    / "study"
    / "school"
    / "gatech"
    / "chappity"
    / "readings"
    / "papers"
)

# Uncomment if you need to calculate embeddings
loader = PyPDFDirectoryLoader(docs_dir)
docs = loader.load()
text_splitter = SemanticChunker(embeddings=OllamaEmbeddings(model=EMBEDDING_MODEL))
splits = text_splitter.split_documents(docs)

persist_directory = ".chroma"
# Uncomment if you need to calculate embeddings
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
    persist_directory=persist_directory,
)
# vectorstore = Chroma(
#     embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
#     persist_directory=persist_directory,
# )
retriever = vectorstore.as_retriever()

rag_prompt = hub.pull("rlm/rag-prompt")

llm = ChatOllama(model=INFERENCE_MODEL, temperature=0, verbose=True)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("AI: How can I help you today?")
while True:
    user_input = input(">>> ").strip().lower()
    if user_input in ["exit", "exit()"]:
        print("Bye!")
        exit(0)

    if user_input:
        try:
            response = rag_chain.invoke(user_input)
            print(response)
        except Exception as e:
            print("Error:", str(e))
