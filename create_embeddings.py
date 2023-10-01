import pickle
from sentence_transformers import SentenceTransformer

skill_list_file = r'./master_skills_list.txt' # skill names from ChatGPT
skill_emb_file = r'./master_emb_list.pkl' # output file
model_path = r'./model' # the path to the downloaded model

# use the model we downloaded in the model directory
model = SentenceTransformer(model_path)

# read in the skill names
with open(skill_list_file, 'r') as f:
    lines = f.readlines()
    master_skills_list = []
    for l in lines:
        master_skills_list.append(l.replace("\n", ""))

# create the embeddings and write it as a pickle file
master_skill_embs = model.encode(master_skills_list)
with open(skill_emb_file, 'wb') as f:
    pickle.dump(master_skill_embs, f)