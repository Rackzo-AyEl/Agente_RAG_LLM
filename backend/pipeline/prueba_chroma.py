import chromadb
import torch
from sentence_transformers import SentenceTransformer


# Crear base de datos
cliente = chromadb.PersistentClient(
    path="./chroma_db"
)

coleccion = cliente.get_or_create_collection(name='documentos')

datos = coleccion.get(
    limit=1,
    include=["documents", "embeddings", "metadatas"]
)

pregunta = "Qué es el perceptrón?"


# Comprobar disponibilidad de dispositivo para vectorizar
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Crear modelo de embedding
modelo = SentenceTransformer(
    "BAAI/bge-m3",
    device=device
)

vector = modelo.encode(
    pregunta,
    normalize_embeddings=True
)

resultado = coleccion.query(

    query_embeddings=[vector.tolist()],

    n_results=3
)

for distancia, documento in zip(

    resultado["distances"][0],

    resultado["documents"][0]

):

    print("="*80)
    print(distancia)
    print(documento[:500])
