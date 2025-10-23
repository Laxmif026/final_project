from flask import Flask, request, jsonify
from utils.retrieve_doc import DocumentRetriever
from utils.generator import GPTGenerator

app = Flask(__name__)

# Initialize RAG components
try:
    retriever = DocumentRetriever()
    generator = GPTGenerator()
except Exception as e:
    raise RuntimeError(f"Failed to initialize components: {e}")

@app.route('/answer', methods=['POST'])
def answer_question():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query is required"}), 400

        results = retriever.retrieve_documents(query, top_k=3)

        if not results:
            return jsonify({
                "answer": "I don't have that information in the available documents.",
                "sources": []
            })

        context = "\n\n".join([doc.get("content", "") for doc in results])
        answer = generator.generate_answer(query, context)

        if len(answer.split()) < 10:
            relevant_sentences = [
                s for s in context.split(".")
                if any(word in s.lower() for word in query.lower().split())
            ]
            answer = f"Here’s what I found: {relevant_sentences[0].strip()}." if relevant_sentences else \
                     "I don't have that information in the available documents."

        sources = list({doc.get("source", "Unknown") for doc in results})

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)