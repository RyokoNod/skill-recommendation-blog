import pandas as pd
import pypdf
import pickle
import re
from sentence_transformers import SentenceTransformer, util
import torch
from flask import Flask, request, json, Response
from flask_restx import Api, Resource, fields, abort
import werkzeug
import numpy as np

# the Flask app name is 'app'
app = Flask(__name__)
app.json.ensure_ascii = False

# API headers
api = Api(app, version='1.0', title='Skills from CV API',
         description='API to parse CVs and return skill recommendations')

# POST argument parser
file_upload_parser = api.parser()
file_upload_parser.add_argument('file', location='files', type=werkzeug.datastructures.FileStorage, required=True,
                               help='PDF file to be parsed')

# model from huggingface that creates sentence embeddings
embedder = SentenceTransformer('./model')

# response field documentation
recommendation_fields = api.model('Recommendations',{
  'Skill': fields.String(description = "The skill name."),
  'Score': fields.Float(description = "The match score (cosine similarity)."),
})
response_fields = api.model('Response', {
    'recommendations': fields.List(fields.Nested(recommendation_fields),
                       description = "Skill names and match score for the recommended skills.\
                       The recommendations are in descending order by score.")
})

def read_pdf_text(pdf_path):
    '''
    Parses a PDF file and returns the contents.
    '''

    # open PDF and extract text from it
    file_text = ''
    with open(pdf_path, 'rb') as f:
        pdf = pypdf.PdfReader(f)
        for page in range(len(pdf.pages)):
            file_text += (pdf.pages[page].extract_text())

    return file_text

def cut_and_clean(string):
    '''
    Cut up text into smaller pieces for the model to read and clean the pieces.
    '''

    # cut up string into chunks using \n and . as a delimiter
    # throw away the pieces that are shorter than 5 words
    chunks = re.split(r'\n|\.', string)
    chunks = [x for x in chunks if len(x) > 4]

    # clean the chunks
    c_chunks = list()
    for i in chunks:
        i = ''.join((x for x in i if not x.isdigit()))  # throw away digits
        i = re.sub(r'[^a-zA-Z0-9 \n\.,]', ' ', i)  # throw away special characters
        i = " ".join(i.split())  # remove extra spaces
        i = i.lower()  # lowercase
        if len(i.split()) > 3:
            # discard chunks that are shorter than 3 words after cleaning
            c_chunks.append(i)

    return c_chunks

def match_snippets(snippets, master_phrase_embs, master_phrase_list, top_k):
    '''
    Match a list of short phrases to a set of phrase embeddings.
    :param snippets:           A list of snippets you want to match to a master phrase list
    :param master_phrase_embs: An embedded version of the master phrase list. The embedder
                               you used for it should be declared as a global variable
    :param master_phrase_list: A list of the raw strings you want to match with snippets
    :param top_k:              How many suggested matches you want per snippet
    :return:                   A pandas dataframe containing the matches from master_phrase_embs,
                               the cosine similarity of the match, and the snippet that corresponds to the match
    '''


    skill_recommendation = pd.DataFrame()
    for query in snippets:
        # convert snippet to embedding, get cosine similarities between the
        # master skill dataset, get top k most similar results
        query_embedding = embedder.encode(query.strip(), convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, master_phrase_embs)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        # organize the skill names, cosine similarity scores, and snippet in a dataframe
        skills_list = list()
        score_list = list()
        for score, idx in zip(top_results[0], top_results[1]):
            skills_list.append(master_phrase_list[idx])
            score_list.append(score.item())
        skills_df = pd.DataFrame(skills_list)
        score_df = pd.DataFrame(score_list)
        sk_sc_df = pd.concat([skills_df, score_df], axis=1)
        sk_sc_df.columns = ['Phrase', 'Score']
        skill_recommendation = pd.concat([skill_recommendation, sk_sc_df]).reset_index(drop=True)

    return skill_recommendation


# /skills_from_cv is the endpoint where the POST request is submitted
@api.route('/skills_from_cv')
class SkillsFromCV(Resource):
    @api.marshal_with(response_fields, as_list=True)
    @api.expect(file_upload_parser)
    def post(self, top_k=5):
        '''
        Get a set of suggestions for skills from a CV
        '''
        args = file_upload_parser.parse_args()

        # get the CV file from the arguments and save it as file.pdf
        input_file = args['file']
        input_file.save('file.pdf')

        # get the list of skill names and the corresponding binary
        # the Cloud Run app runs in /usr/src/app, and in the Dockerfile we copied the embeddings directly under there
        master_skills_emb_binary = r'./master_emb_list.pkl'
        master_skills_list = r'./master_skills_list.txt'
        with open(master_skills_emb_binary, 'rb') as f:
            master_phrase_embs = pickle.load(f)
        with open(master_skills_list, 'r') as f:
            lines = f.readlines()
            master_phrase_list = []
            for l in lines:
                master_phrase_list.append(l.replace("\n", ""))

        # read the pdf in as text
        file_text = read_pdf_text('file.pdf')

        # get recommendations
        cv_snippets = cut_and_clean(file_text)
        skill_recommendation = match_snippets(cv_snippets,
                                              master_phrase_embs,
                                              master_phrase_list,
                                              top_k=top_k)

        # throw away recommendations with scores under 0.5,
        # sort recommendation by scores, drop duplicate skill suggestions
        skill_recommendation = skill_recommendation[skill_recommendation['Score'] >= 0.5]
        skill_recommendation = skill_recommendation.sort_values('Score', ascending=False)
        skill_recommendation = skill_recommendation.drop_duplicates(subset='Phrase').reset_index(drop=True)
        skill_recommendation = skill_recommendation.rename(columns={'Phrase': 'Skill'})

        skill_recommendation = skill_recommendation.replace({np.nan: None})
        response = {'recommendations': skill_recommendation.to_dict(orient='records')}

        return response

    # no if __name__ == "__main__" here. Flask and Flask-Restx do not use this
