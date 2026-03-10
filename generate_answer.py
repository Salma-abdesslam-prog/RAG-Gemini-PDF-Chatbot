import os
from typing import Dict, List
from google import genai
from google.genai.errors import APIError
from google.genai import types


def generate_rag_answer_from_search(
    search_output: Dict[str, List[str]],
    user_prompt: str,
    llm_model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
    max_output_tokens: int = 512,
) -> str:
    """
    Generate a RAG answer from semantic search output using Gemini.

    Authentication uses the GEMINI_API_KEY environment variable.

    search_output should follow this structure:
        {
            "documents": [...],
            "ids": [...],
            "distances": [...]
        }
    """

    # 1) Initialize Gemini client.
    try:
        client = genai.Client()
    except Exception:
        return "Error: GEMINI_API_KEY is missing or invalid."

    # 2) Validate and prepare context.
    if not search_output or "documents" not in search_output or not isinstance(search_output.get("documents"), list):
        return "Invalid search_output: missing 'documents' key or wrong format."

    context_chunks = search_output["documents"]
    if len(context_chunks) == 0:
        return "No context chunks were found by semantic search."

    context_text = "\n\n".join(context_chunks)

    # 3) Build RAG prompt.
    system_instruction = (
        "You are a helpful assistant. Answer only from the provided context. "
        "If the answer is not in the context, clearly say the information is not available "
        "in the provided sources. Keep responses concise and direct."
    )

    full_prompt = f"""
=== PROVIDED CONTEXT ===
{context_text}

=== USER QUESTION ===
{user_prompt}
"""

    # 4) Generation config.
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    # 5) Gemini API call.
    try:
        response = client.models.generate_content(
            model=llm_model,
            contents=full_prompt,
            config=config,
        )

        if response.text:
            return response.text.strip()
        return "Warning: the model returned an empty or blocked response."

    except APIError as e:
        return f"Gemini API error ({e.status_code}): {e.message}"
    except Exception as e:
        return f"Unexpected error while calling Gemini: {e}"


if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("Please set the GEMINI_API_KEY environment variable.")
    else:
        simulated_search_output: Dict[str, List[str]] = {
            "documents": [
                "Python was created by Guido van Rossum and first released in 1991.",
                "Gemini 2.5 Flash is a multimodal model designed for speed and efficiency.",
            ],
            "ids": ["doc_1", "doc_2"],
            "distances": [0.15, 0.22],
        }

        user_question = "Who created Python and when was it released?"

        print(f"Question: {user_question}")
        print("Generating RAG answer with Gemini 2.5 Flash...")

        answer = generate_rag_answer_from_search(
            search_output=simulated_search_output,
            user_prompt=user_question,
            temperature=0.1,
        )

        print("\n=== FINAL MODEL ANSWER ===")
        print(answer)
        print("==========================")
