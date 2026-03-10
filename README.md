# RAG-Gemini-PDF-Chatbot

An intelligent assistant that answers questions about PDF files.
You can upload a PDF, ask a question, and the chatbot returns an answer generated from the document content.

---

## Features
- Requires a Google AI Studio API key (Gemini)
- PDF upload
- Ask a question about PDF content
- LLM-generated answer
- Local or cloud execution, depending on your setup
- Multi-PDF support

---

## Prerequisites

- Python 3.11 or higher
- pip

---

## Installation

1. Clone the project:

```bash
git clone https://github.com/Salma-abdesslam-prog/RAG-Gemini-PDF-Chatbot.git
cd RAG-Gemini-PDF-Chatbot
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

3. Activate the virtual environment:

```bash
# Windows PowerShell
venv\Scripts\Activate.ps1

# Linux/macOS
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Run the application:

```bash
streamlit run app.py
```

2. Open the browser URL shown by Streamlit (default: http://localhost:8501).

3. Upload your PDF file in the interface.

4. Ask your question.

5. Read the generated answer.

---

## Example

The screenshot below shows the chatbot in action. The PDF used in the example explains RAG (Retrieval-Augmented Generation). The question asked is about what RAG is and its benefits. The chatbot returns an answer using the most relevant retrieved passages.

<img width="2495" height="1344" alt="chatbot_cap" src="https://github.com/user-attachments/assets/d74450e4-9b0c-4fe1-9d8e-948677187ff0" />
