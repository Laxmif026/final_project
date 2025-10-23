#!/usr/bin/env python3
"""
Generator Module
================
Handles the GENERATION part of RAG:
- Takes user query and retrieved context
- Uses Azure OpenAI GPT model to generate accurate answers
- Applies prompt engineering and fallback logic

Author: HR RAG Bot Team
Date: October 2025
"""

from openai import AzureOpenAI
from utils.setting import (
    AZURE_OPENAI_CHAT_ENDPOINT,
    AZURE_OPENAI_CHAT_API_KEY,
    CHAT_MODEL_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION
)

class GPTGenerator:
    """Generates answers using Azure OpenAI GPT model based on retrieved context."""

    def __init__(self):
        print(" Initializing GPT Generator...")
        self.chat_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
            api_key=AZURE_OPENAI_CHAT_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        print(" GPT Generator initialized successfully!\n")

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using GPT model based on context.
        - If answer exists: descriptive response (2–3 sentences)
        - If partial info: summarize available details
        - If nothing relevant: say 'I don't have that information in the available documents.'
        """
        if not context or "No relevant documents" in context:
            return "I don't have that information in the available documents."

        # Short, effective prompt
        system_prompt = (
            "You are an HR assistant. Answer based ONLY on the provided context. "
            "If the answer exists, give a clear, descriptive response in 2–3 sentences. "
            "If nothing relevant is found, say: 'I don't have that information in the available documents.' "
            "Do not guess or invent details."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        try:
            response = self.chat_client.chat.completions.create(
                model=CHAT_MODEL_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Keep answers factual
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {e}"