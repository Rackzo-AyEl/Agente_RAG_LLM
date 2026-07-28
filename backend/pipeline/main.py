# Librerías
import os
import torch
import warnings
import logging
from dotenv import load_dotenv

# Importaciones de LangChain
from inferencia import inferencia_RAG
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


# Silenciar advertencias de HuggingFace y LangChain
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")


""" Inicio de programa """
if __name__ == '__main__':
    # Configuración inicial y variables de entorno
    load_dotenv()
    if not os.getenv('GEMINI_API_KEY'):
        raise ValueError("No se pudo cargar la API KEY de Gemini")

    # Configuración del sistema
    USAR_MULTIQUERY = False
    K_DOCUMENTOS = 3
    NIVEL_TEMPERATURA = 0.3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Instanciar el retriever
    print("\n[*] Cargando modelos y vaiables de sistema...")
    modelo_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Enlazar con la base de datos vectorial
    vectorstore = Chroma(
        collection_name="documentos",
        persist_directory="./chroma_db",
        embedding_function=modelo_embeddings
    )

    # Instanciar el LLM de Gemini (necesario para multiQuery)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=NIVEL_TEMPERATURA
    )

    # Configuración del retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": K_DOCUMENTOS})

    if USAR_MULTIQUERY:
        print("\n[-] Iniciando sistema con MultiQueryRetriever activo...")
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
    else:
        retriever = base_retriever

    # Consulta interactiva en consola
    print("\n--- Sistema RAG listo ---")
    while True:
        try:
            pregunta = input("\n> ").strip()
            if not pregunta:
                continue
            if pregunta.lower() in ['salir', 'q']:
                print("¡Hasta luego!")
                break

            inferencia_RAG(llm, pregunta, retriever)


        except (KeyboardInterrupt, EOFError):
            print("\n¡Hasta luego!")
            break


