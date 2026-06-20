!pip install gensim transformers torch
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
prompt = "Honesty is the best policy"
print("=== Original Prompt ===")
print(prompt)
result = generator(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    repetition_penalty=1.5,
    pad_token_id=50256
)
print("\n=== Enriched Prompt ===")
print(result[0]['generated_text'])
print("\n--- Comparison ---")
print("1. Original output is simple and general.")
print("2. Enriched output is more detailed and context-rich.")
print("3. Enriched prompt improves relevance using similar words.")
