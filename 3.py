!pip install gensim matplotlib scikit-learn
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
corpus = [
"The patient was diagnosed with diabetes and hypertension.",
"MRI scans reveal abnormalities in the brain tissue.",
"The treatment involves antibiotics and regular monitoring.",
"Symptoms include fever, fatigue, and muscle pain.",
"The vaccine is effective against several viral infections.",
"Doctors recommend physical therapy for recovery.",
"The clinical trial results were published in the journal.",
"The surgeon performed a minimally invasive procedure.",
"The prescription includes pain relievers and anti-inflammatory drugs.",
"The diagnosis confirmed a rare genetic disorder."
]
data = [s.lower().split() for s in corpus]
model = Word2Vec(data, vector_size=100,window=5, min_count=1,epochs=50)
words = list(model.wv.index_to_key)
emb = np.array([model.wv[w] for w in words])
result = TSNE(n_components=2,random_state=42,perplexity=5).fit_transform(emb)
plt.figure(figsize=(10,8))
for i,w in enumerate(words):
    plt.scatter(result[i,0], result[i,1])
    plt.text(result[i,0], result[i,1], w)
plt.title("Word Embeddings Visualization (Medical Domain)")
plt.show()
print("Words similar to 'treatment':")
for w,s in model.wv.most_similar("treatment", topn=5):
    print(f"{w} ({s:.2f})")
print("\nWords similar to 'vaccine':")
for w,s in model.wv.most_similar("vaccine", topn=5):
    print(f"{w} ({s:.2f})")
