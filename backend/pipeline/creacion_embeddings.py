# Librerías
# import pymupdf4llm
# import re
# import unicodedata
import torch
from extraccion_pdfs_md import extraccion_pdf_md
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# Función para generar chunks
def generacion_chunks(md, source):
    headers = [
        ("#", "h1"),
        ("##", "h2")
    ]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False
    )

    documentos = header_splitter.split_text(md)

    # Filtrar documentos pequeños que no aportan
    documentos = [doc for doc in documentos if len(doc.page_content.strip()) > 100]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documentos)

    for i, chunk in enumerate(chunks):
    
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = source

    # print(len(documentos))
    #
    # for doc in documentos:
    #     print(len(doc.page_content))
    #
    # lista_chunks = [i for i in enumerate(chunks)]
    # for i, chunk in lista_chunks[:10]:
    #
    #     print("=" * 80)
    #     print(f"Chunk {i}")
    #     print(chunk.metadata)
    #     print("-" * 80)
    #     print(chunk.page_content[:1000])

    return chunks


def probar_modelo(nombre_modelo, chunks):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nCargando {nombre_modelo}")
    
    modelo = SentenceTransformer(
        nombre_modelo,
        device=device
    )

    textos = [
        chunk.page_content 
        for chunk in chunks
    ]

    embeddings = modelo.encode(
        textos,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Dimensión:")
    print(embeddings.shape)

    return modelo, embeddings


# Inicio de programa
if __name__ == '__main__':
    # Ruta de archivos
    ruta_principal = Path('../../data/')

    # Comprobar disponibilidad de gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Iterar por cada PDF en la carpeta de datos
    for pdf in ruta_principal.glob('*.pdf'):
        print(f"Procesando archivo: {pdf.name}")

        try:
            # Llamada a extracción de MD
            print("Extrayendo con extraccion_pdf_md")
            md_pdf = extraccion_pdf_md(pdf)

            # Llamada a generación de chunks
            print("Generando chunks con generacion_chunks")
            chunks = generacion_chunks(md_pdf, pdf.name)

            chunks_prueba = chunks[:100]

            print("Probando modelo con probar_modelo")
            modelo, embeddings = probar_modelo("BAAI/bge-m3", chunks)

            pregunta = "¿Que son las listas descriptivas?"
            
            vector_pregunta = modelo.encode(
                pregunta,
                normalize_embeddings=True
            )
            
            
            similitudes = cos_sim(
                vector_pregunta,
                embeddings
            )[0]
            
            
            indices = similitudes.argsort(
                descending=True
            )[:5]
            
            
            for i in indices:
            
                print("="*80)
                print(similitudes[i])
                print(chunks[i].page_content[:500])

        except Exception as e:
            print(f"Error en el procesamiento de {pdf.name}\nError: {e}")
