from sentence_transformers import SentenceTransformer

# download pretrained
model = SentenceTransformer('all-MiniLM-L6-v2')

# save to local directory
model.save('./model/')