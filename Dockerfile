#FROM python:3.10
FROM python:3.10-slim-bullseye

# make sure every file you need is inside the working dir of container
WORKDIR /usr/src/app
COPY main.py /usr/src/app/main.py
COPY model /usr/src/app/model
COPY master_emb_list.pkl /usr/src/app/master_emb_list.pkl
COPY master_skills_list.txt /usr/src/app/master_skills_list.txt

RUN apt-get -qqy update && apt-get install -qqy

# Install  dependencies. Here ./ refers to WORKDIR, which is /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# get start up file in place
COPY helper_script.sh /usr/src/app/helper_script.sh
RUN chmod +x helper_script.sh

# run start up file
CMD ["/usr/src/app/helper_script.sh"]
