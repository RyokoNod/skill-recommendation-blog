#TRANSLATE_KEY = r'./translate_key.json'

def noneng_language(string):
  '''
  Returns True when the input string may not be English. The language detectors
  have a bit of randomness so the results may not always be the same.
  '''
  import pandas as pd
  import langdetect
  import langid
  import gcld3
  import re
  
  detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
  
  # get detection results from different libraries
  langid_result = langid.classify(string)[0]
  langdetect_result = langdetect.detect(string)
  gcld_result = detector.FindLanguage(text=string).language
  
  # get majority vote for the detection results. Chinese is just defined as zh
  lang_votes = [langid_result, langdetect_result, gcld_result]
  lang_votes = list(map(lambda x: re.sub(r'zh-.*', r'zh',x), lang_votes))
  
  if all(vote != 'en' for vote in lang_votes):
    return True
  else:
    return False
  

def google_translate(text, target="en"):
  '''
  Uses Google Translate API to translate a string to the target language (does
  not do lists or pandas.Series).
  
  <Inputs>
  text: String of text to translate.
  target: The target language. Default is English.
  
  <Returns>
  The translated version of the input string.
  '''
  import os
  import six
  from google.cloud import translate_v2 as translate
  
  #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = TRANSLATE_KEY

  if isinstance(text, six.binary_type):
    text = text.decode("utf-8")
  
  translate_client = translate.Client()
  
  result = translate_client.translate(text, target_language=target)
  
  return result
