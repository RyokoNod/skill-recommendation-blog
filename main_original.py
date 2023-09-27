from flask import Flask, request, json, Response
from flask_restx import Api, Resource, fields, abort
import pandas as pd
from translate_text import noneng_language, google_translate
import werkzeug
import os
import pypdf
import pickle
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util 
from torch import topk

# the Flask app name is 'app'. The main function of app is marked with @app
app = Flask(__name__)
app.json.ensure_ascii = False

# API headers
api = Api(app, version='1.0', title='Skills from CV API',
         description='API to parse CVs and return skill recommendations and the language of the CV')

# POST argument parser
tenant_set = ('accent', 'nipro', 'ravago', 'ngage', 'skillbuilders','sdworx', 'protime')
env_set = ('environment1', 'environment2', 'environment3')
file_upload_parser = api.parser()
file_upload_parser.add_argument('file', location='files', type=werkzeug.datastructures.FileStorage, required=True,
                               help='PDF file to be parsed')
file_upload_parser.add_argument('tenant', type=str, required=True, help='Tenant name', 
                                choices=tenant_set)
file_upload_parser.add_argument('environment', type=str, required=True, help='Environment information',
                                choices=env_set)

# response field documentation
recommendation_fields = api.model('Recommendations',{
  'de': fields.String(description = "The skill name in German."),
  'en': fields.String(description = "The skill name in English. This will always be returned."),
  'es': fields.String(description = "The skill name in Spanish."),
  'fr': fields.String(description = "The skill name in French."),
  'hu': fields.String(description = "The skill name in Hungarian."),
  'it': fields.String(description = "The skill name in Italian."),
  'nl': fields.String(description = "The skill name in Dutch."),
  'pl': fields.String(description = "The skill name in Polish."),
  'pt': fields.String(description = "The skill name in Portuguese."),
  'sr': fields.String(description = "The skill name in Serbian."),
  'tr': fields.String(description = "The skill name in Turkish."),
  'zh': fields.String(description = "The skill name in Chinese."),
  'score': fields.Float(description = "The match score (cosine similarity)."),
  'value_list_item_id': fields.Integer(description = "The value_list_item_id. Not null when the skill exists as a value \
                                                      in the tenant's DB."),
})
response_fields = api.model('Response', {
    'cvlang': fields.String(description = "The language of the CV. The language code is the one returned from Cloud Translate API (Basic)."),
    'recommendations': fields.List(fields.Nested(recommendation_fields), 
                                   description = "Skill names, value list item ids, and match score for the recommended skills.\
                                                  The recommendations are in descending order by score. The languages other than \
                                                  English will be null if the skill already exists in the tenant DB or there is \
                                                  no translation provided in the skill list from huapii or customer.")
})

#@api.errorhandler(510)
#def pdf_parser_failed(error):
#  return {'message': 'Test message'}, 510

# model from huggingface that creates sentence embeddings
embedder = SentenceTransformer('./model')

# where the embeddings for each environment and tenant are kept
# path to single mounted Cloud Bucket
embeddings_path = r'/home/skill_list/'

def read_pdf_text(pdf_path, write_contents=False):
  '''
  Parses a PDF file and returns the contents. Writes out the parse results
  if specified.
  '''
  
  # open PDF and extract text from it
  file_text = ''
  with open(pdf_path, 'rb') as f:
    pdf = pypdf.PdfReader(f)
    for page in range(len(pdf.pages)):
      file_text += (pdf.pages[page].extract_text())
  
  # write parse results as parse_results.txt
  output_file = r"./parse_results.txt"
  if write_contents == True:
    with open(output_file, "w") as f:
      print(file_text, file=f)
  
  return file_text

def cut_and_clean(string):
  '''
  Cut up text into smaller pieces for the model to read and clean the pieces.
  '''
  
  # cut up string into chunks using \n and . as a delimiter
  # throw away the pieces that are shorter than 5 words
  chunks = re.split(r'\n|\.', string)
  c_chunks = [x for x in chunks if len(x) > 4]
  
  # clean the chunks
  c_chunks = list()
  for i in chunks:
    i = ''.join((x for x in i if not x.isdigit())) # throw away digits
    i = re.sub(r'[^a-zA-Z0-9 \n\.,]', ' ', i) # throw away special characters 
    i = " ".join(i.split()) # remove extra spaces
    i = i.lower() # lowercase
    if len(i.split()) > 3:
      # discard chunks that are shorter than 3 words after cleaning
      c_chunks.append(i) 
  
  return c_chunks

def match_snippets(snippets, master_phrase_embs, master_phrase_df, top_k):
  '''
  Match a list of short phrases to a set of phrase embeddings.
  
  <Inputs>
  snippets: A list of phrases you want to match to a master phrase list
  master_phrase_embs: An embedded version of the master phrase list. The embedder
                      you used for it should be declared as a global variable
  master_phrase_df: A dataframe of the skill names and value list item ids 
                    you want to match with snippets
  top_k: How many suggested matches you want per snippet
  
  <Output>
  A pandas dataframe containing the matches from master_phrase_embs,
  the cosine similarity of the match, and the snippet that corresponds to the match
  '''
  
  skill_recommendation = pd.DataFrame()
  for query in snippets:
    # convert snippet to embedding, get cosine similarities between the
    # master skill dataset, get top k most similar results
    query_embedding = embedder.encode(query.strip(), convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, master_phrase_embs)[0]
    top_results = topk(cos_scores, k=top_k)

    # organize the skill names, cosine similarity scores, and snippet in a dataframe
    topk_scores = top_results[0].squeeze().tolist()
    topk_indices = top_results[1].squeeze().tolist()

    snippet_recommendation =  master_phrase_df.iloc[topk_indices, :].copy()
    snippet_recommendation['score'] = topk_scores
    skill_recommendation = pd.concat([skill_recommendation, snippet_recommendation])
  
  return skill_recommendation


# /skills_from_cv is the endpoint where the POST request is submitted
# in this case the POST request is the CV's file name
@api.route('/skills_from_cv')
class SkillsFromCV(Resource):
  @api.marshal_with(response_fields)
  @api.expect(file_upload_parser)
  @api.response(510, "PDF parser failed")
  def post(self, top_k=5, write_parse_results=False):
    '''
    Get a set of suggestions for skills from a CV
    '''
    args = file_upload_parser.parse_args()

    # the POST request has the key-value pair {cv_name: 'CV file name'}
    input_file = args['file']
    input_file.save('file.pdf')
    environment = args['environment']
    tenant = args['tenant']
    
    # get the list of skill names and the corresponding binary
    master_skills_emb_binary = os.path.join(embeddings_path, environment, tenant, 'skill_name_embeddings')
    master_skills_df_csv = os.path.join(embeddings_path, environment, tenant, 'skill_df.csv')
    with open(master_skills_emb_binary, 'rb') as f:
      master_phrase_embs = pickle.load(f)
    master_phrase_df = pd.read_csv(master_skills_df_csv)
    
    # if the cv was written in a language other than English, translate it first
    try:
      file_text = read_pdf_text('file.pdf', write_contents=write_parse_results)
    except:
      abort(510)
    file_lang = 'en'
    if noneng_language(file_text) == True:
      translation_results = google_translate(file_text)
      file_text = translation_results['translatedText']
      file_lang = translation_results['detectedSourceLanguage']
    
    # get recommendations
    cv_snippets = cut_and_clean(file_text)
    skill_recommendation = match_snippets(cv_snippets, 
                                          master_phrase_embs, 
                                          master_phrase_df, 
                                          top_k = top_k)

    # throw away recommendations with scores under 0.5,
    # sort recommendation by scores, drop duplicate skill suggestions
    skill_recommendation = skill_recommendation[skill_recommendation['score']>=0.5]
    skill_recommendation = skill_recommendation.sort_values('score', ascending=False)
    skill_recommendation = skill_recommendation.drop_duplicates(subset='en').reset_index(drop=True)
    
    skill_recommendation = skill_recommendation.replace({np.nan:None})
    response = {'recommendations': skill_recommendation.to_dict(orient='records'),
                'cvlang': file_lang}
    
    return response

  # no if __name__ == "__main__" here. Flask and Flask-Restx do not use this


