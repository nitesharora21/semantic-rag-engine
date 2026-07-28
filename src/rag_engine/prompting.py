from rag_engine.models import Chunk

def build_rag_prompt(
    question: str,
    chunks: list[Chunk],
    ) -> str:
    """
    Takes in the user prompted question and all the semantically close chunks retrieved using FAISS Index and constructing a context and rule based response to feed it into LLM.
    """
    context_sections = []
    for index, chunk in enumerate(chunks, start=1):
        context_sections.append(
            f"[Source {index}]\n"
            f"\tChunk ID: {chunk.id}\n"
            f"\tSource: {chunk.source}\n"
            f"\tContent:\n\t\t{chunk.text}\n"
            f"{'*' * 50}"
        )
    # Takes all the most relevant chunks AND creates a flattened string of the Chunk MetaData and its content (text)
    context = "\n\n".join(context_sections)
    return f""" You are question-answering assistant:
    Answer using the user's question from the provided context only.
    Follow these rules strictly:
    1. Please dont use outside knowledge
    2. Do not create or invent new information outside of the given prompts
    3. Acknowledge if the context does not have the necessary information and say: "Me dont know how me can tell ya anything, try something else bud"
    4. Ensure to cite sources using the format [Source 1], [Source 2], etc.
    5. Keep the answers specific, sharp and precise.

    Context:
    {context}

    Question:
    {question}

    Answer:
        
    """

