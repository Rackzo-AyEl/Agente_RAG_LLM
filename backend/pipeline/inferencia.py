# Librerías
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


""" Concatena el contenido de los documentos recuperados """
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


""" Función para generar inferencia """
def inferencia_RAG(llm, pregunta, retriever):
    # Definir el prompt del sistema y usuario
    system_prompt = """Eres un asistente virtual experto. Responde a la pregunta del usuario utilizando únicamente la siguiente información de contexto recuperada.
Si la respuesta no se encuentra en el contexto, indica claramente que no posees suficiente información en los documentos cargados.
Entrega el contenido en formato MarkDown

Contexto: {context}"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # Construir la cadena RAG usando LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Recuperar fuentes directamente para mostrarlas
    # docs_recuperados = retriever.invoke(pregunta)
    # print(f"\nDEBUG: Documentos recuperados: {len(docs_recuperados)}")
    # for doc in docs_recuperados:
    #     print(f"Contenido: {doc.page_content[:50]}...")

    # Generar respuesta usando la cadena
    respuesta = rag_chain.invoke(pregunta)

    print("\n" + "=" * 30)
    print("RESPUESTA:")
    print("=" * 30)
    print(respuesta)
