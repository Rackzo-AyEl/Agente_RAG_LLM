#NOTE: Código para extraer datos crudos de PDFs, preprocesarlos y devolver contenido en formato MD

# Librerías
import pymupdf4llm
import re
import unicodedata
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pathlib import Path


# Función de extracción de contenido de PDF
def extraccion_pdf_md(pdf_entrada):

    md = pymupdf4llm.to_markdown(pdf_entrada)

    # Normalizar unicode
    md = unicodedata.normalize("NFC", md)

    # Unificar saltos de línea
    md = md.replace("\r\n", "\n").replace("\r", "\n")

    # Eliminar caracteres invisibles frecuentes
    md = (
        md.replace("\ufeff", "")
          .replace("\u200b", "")
          .replace("\u200c", "")
          .replace("\u200d", "")
    )

    md = re.sub(r"<[^>]+>", "", md)

    # ![](imagen.png)
    md = re.sub(r"!\[.*?\]\(.*?\)", "", md)

    md = re.sub(
        r"https?://\S+\.(?:png|jpg|jpeg|gif|bmp|svg|webp)",
        "",
        md,
        flags=re.IGNORECASE,
    )

    lineas_limpias = []
    dentro_codigo = False

    for linea in md.split("\n"):

        # Detectar bloques de código
        if linea.strip().startswith("```"):
            dentro_codigo = not dentro_codigo
            lineas_limpias.append(linea)
            continue

        if dentro_codigo:
            lineas_limpias.append(linea)
            continue

        # Quitar espacios al inicio/final
        linea = linea.rstrip()

        # No modificar tablas Markdown
        if "|" not in linea:
            linea = re.sub(r"[ \t]{2,}", " ", linea)

        lineas_limpias.append(linea)

    md = "\n".join(lineas_limpias)

    # Máximo dos líneas vacías consecutivas
    md = re.sub(r"\n{3,}", "\n\n", md)

    return md


