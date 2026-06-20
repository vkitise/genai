!pip install gensim
import gensim.downloader as api
import random
model = api.load("glove-wiki-gigaword-100")
def create_paragraph(seed):
    words = [w for w,_ in model.most_similar(seed, topn=10)]
    random.shuffle(words)
    words = words[:5]
    return (
        f"In a world defined by {seed}, people found themselves "
        f"surrounded by concepts like {', '.join(words[:-1])}, "
        f"and {words[-1]}. These ideas shaped the way they thought, "
        f"acted, and dreamed. Every step forward reflected the essence "
        f"of '{seed}', bringing them closer to understanding "
        f"the true meaning of {words[0]}."
    )
print(create_paragraph("freedom"))
