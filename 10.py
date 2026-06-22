!pip install sentence-transformers faiss-cpu pdfplumber
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
def extract_text(pdf):
    text = ""
    with pdfplumber.open(pdf) as p:
        for page in p.pages:
            if page.extract_text():
                text += page.extract_text()
    return text
text = extract_text("ipc.pdf")
chunks = [
    " ".join(text.split()[i:i+200])
    for i in range(0, len(text.split()), 200)
]
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("Welcome to IPC Chatbot")
print("Type 'exit' to stop")
while True:
    q = input("You: ")
    if q.lower() == "exit":
        break
    q_emb = model.encode([q])_, idx = index.search(np.array(q_emb),3 )
    print("\nRelevant IPC Sections:\n")
    for i in idx[0]:
        print(chunks[i], "\n")
