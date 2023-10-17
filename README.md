# Making an API out of a Hugging Face model

This is a repository linked to the blog posts written by [Datamarinier](https://datamarinier.be/) and [Huapii](https://huapii.com/). 
The posts were written by data scientists from both Datamarinier and Huapii, the main code was provided from Huapii, and the infrastructure on Google Cloud Platform was designed by Datamarinier.

Here are some notes on what the files and folders contain, but for more information on how to use them, do take a look at our blog ;)

* `model`: The model from Hugging Face
* `Dockerfile`: The docker container settings for deployment on Cloud Run
* `create_embeddings.py`: The script to create embeddings from master_skills_list.txt
* `download_huggingface_model.py`: The script to download the model in the model folder
* `helper_script.sh`: A helper script to accompany Dockerfile
* `main.py`: The main function when deploying to Cloud Run
* `master_emb_list.pkl`: The embeddings created with create_embeddings.py
* `master_skills_list.txt`: The skill list to recommend from
* `recommend_without_cloudrun.py`: The main function written to run without Cloud Run
* `requirements.txt`: The list of Python modules that are needed to run the code
