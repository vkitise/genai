!pip install transformers==4.37.2
from transformers import pipeline
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)
text = """
Artificial Intelligence (AI) is transforming many industries by enabling machines
to learn from data and perform tasks that usually require human intelligence.
AI technologies such as machine learning, natural language processing and
computer vision are widely used in healthcare, finance, education and transportation.
In healthcare AI helps doctors diagnose diseases more accurately.
In finance it is used for fraud detection and risk management.
However, AI also raises ethical concerns such as privacy issues and bias.
"""
summary = summarizer(
    text,
    max_length=60,
    min_length=25,
    do_sample=False
)
print("Summary:")
print(summary[0]["summary_text"])
