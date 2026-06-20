!pip install transformers torch
from transformers import pipeline
sentiment = pipeline("sentiment-analysis")
sentences = [
    "The new phone I bought is absolutely amazing!",
    "Worst customer service ever. I'm never coming back.",
    "The experience was average, nothing special.",
    "Fast delivery and the packaging was perfect.",
    "The product broke within two days. Very disappointed."
]
results = sentiment(sentences)
print("Sentiment Analysis Results:\n")
for s, r in zip(sentences, results):
    print("Input Sentence:", s)
    print(f"Predicted Sentiment: {r['label']}, Confidence Score: {r['score']:.2f}\n")
