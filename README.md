# RAG-Gemini-PDF-Chatbot

Un assistant intelligent capable de répondre à des questions sur des fichiers PDF.  
Vous pouvez **uploader un PDF**, poser une question, et le chatbot vous renvoie une réponse générée à partir du contenu du document.

---

## Fonctionnalités

- 📄 Upload de fichiers PDF  
- 💬 Poser une question (prompt) sur le contenu du PDF  
- 🤖 Réponse générée via un modèle LLM  
- ⚡ Exécution locale ou cloud selon votre configuration  
- 🔄 Multi-PDF supporté pour combiner plusieurs documents  

---

## Prérequis

- Python 3.11 ou supérieur  
- pip

---

## Installation

1. Cloner le projet :

```bash
git clone https://github.com/Salma-abdesslam-prog/RAG-Gemini-PDF-Chatbot.git
cd RAG-Gemini-PDF-Chatbot
```
2. Créer un environnement virtuel (optionnel mais recommandé) :
```bash
python -m venv venv
```
3. Activer l'environnement virtuel :
```bash
source venv/bin/activate
```
4. Installer les dépendances nécessaires :
```bash
pip install -r requirements.txt
```

---

## Usage

1. Lancer l'application :
   
```bash
streamlit run app.py
```
2. Ouvrir le navigateur sur l’URL indiquée (par défaut http://localhost:8501).

3. Uploader votre fichier PDF via l’interface.

4. Poser votre question dans le champ prévu à cet effet.

5. Recevoir la réponse générée par le modèle.




