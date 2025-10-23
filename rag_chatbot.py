from utils.retrieve_doc import DocumentRetriever
from utils.generator import GPTGenerator

retriever = DocumentRetriever()
generator = GPTGenerator()

def answer_question(query: str) -> str:
    results = retriever.retrieve_documents(query, top_k=3)
    if not results:
        return "I don't have that information in the available documents."

    context = "\n\n".join([doc.get("content", "") for doc in results])
    answer = generator.generate_answer(query, context)

    # If GPT response is empty or too short, fallback to relevant sentence
    if len(answer.split()) < 10:
        relevant_sentences = [s for s in context.split(".") if any(word in s.lower() for word in query.lower().split())]
        if relevant_sentences:
            answer = f"Here’s what I found: {relevant_sentences[0].strip()}."
        else:
            answer = "I don't have that information in the available documents."

    sources = "\n\n**Sources:**\n" + "\n".join([doc.get("source", "Unknown") for doc in results])
    return answer + sources