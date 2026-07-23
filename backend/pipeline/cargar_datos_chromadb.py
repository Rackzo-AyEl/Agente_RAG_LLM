# Librerías
import chromadb
import torch
from creacion_embeddings import generacion_chunks, generar_embedding
from extraccion_pdfs_md import extraccion_pdf_md
from pathlib import Path
from sentence_transformers import SentenceTransformer


# Inicio de programa
if __name__ == '__main__':
    ruta_principal = Path('../../data/')

    # Comprobar disponibilidad de dispositivo para vectorizar
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        try:
            # Llamada a extracción de MD
            print("Extrayendo con extraccion_pdf_md")
            md_pdf = extraccion_pdf_md(pdf)

            # Llamada a generación de chunks
            print("Generando chunks con generacion_chunks")
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

            
        except Exception as e:
            print(f"Error en el procesamiento de {pdf.name}\nError: {e}")

