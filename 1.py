!pip install gensim scipy
import gensim.downloader as api
from scipy.spatial.distance import cosine
model = api.load("word2vec-google-news-300")
print("First 10 dimensions of king:")
print(model['king'][:10])
print("\nTop 10 words similar to king:")
for w,s in model.most_similar('king'):
    print(f"{w}: {s:.4f}")
r = model.most_similar(positive=['king','woman'],negative=['man'],topn=1)
print("\nking - man + woman ≈")
print(f"{r[0][0]} : {r[0][1]:.4f}")
print("\nparis + italy - france ≈")
for w,s in model.most_similar(positive=['paris','italy'],negative=['france']):
    print(f"{w}: {s:.4f}")
print("\nwalking + swimming - walk ≈")
for w,s in model.most_similar(positive=['walking','swimming'],negative=['walk']):
    print(f"{w}: {s:.4f}")
print("\nCosine Similarity:")
print(f"{1-cosine(model['king'],model['queen']):.4f}")
