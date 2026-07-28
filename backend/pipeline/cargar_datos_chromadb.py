# Librerías
import chromadb
import torch
from creacion_embeddings import generacion_chunks, generar_embedding
from extraccion_pdfs_md import extraccion_pdf_md
from pathlib import Path
from sentence_transformers import SentenceTransformer


""" Función de carga de contenido a Chromadb
La función retorna True o False para manejo de errores """
def cargar_datos_chomadb(modelo, pdf):
    try:
        # Llamada a extracción de MD
        md_pdf = extraccion_pdf_md(pdf)

        # Llamada a generación de chunks
        chunks = generacion_chunks(md_pdf, pdf.name)

        # Llamada a generación de embeddings
        modelo, embeddings = generar_embedding(chunks, modelo)

        ids = []
        documentos = []
        metadata = []

        for chunk in chunks:
            ids.append(
                f"{chunk.metadata['source']}_{chunk.metadata['chunk_id']}"
            )

            documentos.append(chunk.page_content)

            metadata.append(chunk.metadata)

        # Guardar embedding en Chromadb
        coleccion.add(
            ids=ids,
            documents=documentos,
            embeddings=embeddings.tolist(),
            metadatas=metadata
        )

        # Liberar variables
        del md_pdf
        del chunks
        del embeddings
        del ids
        del documentos
        del metadata

        return True


    # Manejo de errores
    except Exception:
        return False


""" Inicio de programa """
if __name__ == '__main__':
    ruta_principal = Path('../../data/')

    # Comprobar disponibilidad de dispositivo para vectorizar
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando {device} para procesamiento de chunks")

    # Crear base de datos
    cliente = chromadb.PersistentClient(
        path="./chroma_db"
    )

    coleccion = cliente.get_or_create_collection(name='documentos')

    # Crear modelo de embedding
    modelo = SentenceTransformer(
        "BAAI/bge-m3",
        device=device
    )

    # Función para cargar todos los PDFs en Chromadb
    for pdf in ruta_principal.glob('*.pdf'):
        print(f"\n[*] Procesando archivo: {pdf.name}")
        cargar_datos_chomadb(modelo, pdf)

