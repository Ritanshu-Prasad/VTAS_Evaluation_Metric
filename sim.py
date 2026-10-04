
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer('all-MiniLM-L6-v2')
emb1 = model.encode('a woman', convert_to_tensor=True)
emb2 = model.encode('a person', convert_to_tensor=True)
emb3 = model.encode('woman', convert_to_tensor=True)
emb4 = model.encode('person', convert_to_tensor=True)
print('a woman vs a person:', cos_sim(emb1, emb2).item())
print('woman vs person:', cos_sim(emb3, emb4).item())

emb5 = model.encode('a kid', convert_to_tensor=True)
print('a kid vs a person:', cos_sim(emb5, emb2).item())


