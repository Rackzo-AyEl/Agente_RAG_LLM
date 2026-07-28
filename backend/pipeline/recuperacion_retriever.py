from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Función de extracción de embeddings con retriever
def retriever_recuperacion(consulta, device):

    # 1. Configurar el modelo de embeddings
    # LangChain envuelve a SentenceTransformers por detrás
    modelo_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. Conectar a tu base de datos Chroma existente
    # En lugar de usar el cliente nativo de chromadb, usamos el wrapper de LangChain
    vectorstore = Chroma(
        collection_name="documentos",
        persist_directory="./chroma_db",
        embedding_function=modelo_embeddings
    )
    
    # 3. ¡LA MAGIA! Convertir la base de datos en un Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3} # Equivalente a tu n_results=3
    )
    
    # invoke() hace el encode, busca en Chroma, desempaqueta el diccionario y devuelve objetos Document
    documentos_recuperados = retriever.invoke(consulta)

    return documentos_recuperados

