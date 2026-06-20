!pip install gensim matplotlib scikit-learn
import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
print("Loading Word2Vec model...")
model = api.load("word2vec-google-news-300")
words = ['computer','internet','software','hardware',
         'keyboard','mouse','server','network',
         'programming','database']
vectors = [model[w] for w in words]
reduced = PCA(n_components=2).fit_transform(vectors)
print("Top 5 words similar to 'computer':")
for word, score in model.most_similar('computer', topn=5):
    print(f"{word}: {score:.4f}")
plt.figure(figsize=(8,6))
for i, word in enumerate(words):
    plt.scatter(reduced[i,0], reduced[i,1])
    plt.annotate(word, (reduced[i,0], reduced[i,1]))
plt.title("PCA Visualization of Technology Word Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
