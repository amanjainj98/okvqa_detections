import json
import os
import itertools

from nltk.tokenize import word_tokenize
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from spellchecker import SpellChecker
from functools import cmp_to_key
import wikipedia

import multiprocessing 


count = 0
detected_objects = {}

table = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))
spell = SpellChecker()

coco_detections_file = "detected_coco_objects.json"
imagenet_detections_file = "detected_imagenet_objects.json"
places_detections_file = "detected_places.json"

coco_detections = json.load(open(coco_detections_file))
imagenet_detections = json.load(open(imagenet_detections_file))
places_detections = json.load(open(places_detections_file))


def unique_words(sent,query_tuple):
	return int(query_tuple[0] in sent) + int(query_tuple[1] in sent)


def count_words(sent,query_tuple):
	return sent.count(query_tuple[0]) + sent.count(query_tuple[1])

def compare_fn(a,b):
    if a[0] > b[0]:
        return 0
    elif a[0] == b[0]:
        if a[1] > b[1]:
            return 0
        else:
            return 1
    else:
        return 0

def get_sentences(content, query_tuple):
  content = content.splitlines()
  content = [c for c in content if (c and not c.isspace() and not c.startswith("== "))]
  sentences = [sent for c in content for sent in sent_tokenize(c)]
  sentences = [[unique_words(sent,query_tuple), count_words(sent,query_tuple), sent] for sent in sentences]
  sentences_top_5 = sorted(sentences,key=cmp_to_key(compare_fn))[:5]
  sentences_top_5 = [s[2].lower() for s in sentences_top_5]
  return sentences_top_5



def wikipedia_search(query_tuple):
  global spell

  titles = []
  query = (query_tuple[0] + " " + query_tuple[1]).lower()
  try:
    titles = wikipedia.search(query,results=1)
  except Exception: 
    pass

  if not titles:
    ql = query.split()
    query_sc = " ".join([spell.correction(x) for x in ql])  
    
    try:
      titles = wikipedia.search(query_sc,results=1)
    except Exception: 
      pass
    

  if not titles:
    return None

  title = titles[0]
  try:
    sentences = get_sentences(wikipedia.WikipediaPage(title).content,query_tuple)
  except:
    return None

  return {"title" : title.lower(),"sentences" : sentences}

dataset_file_dir = "train2014_questions_split/"
target_dir = "training_set_articles_data_split/"

def prepare_data(i):
  dataset_file = str(i) + ".json"
  training_set_articles_data_i = json.load(open(target_dir + dataset_file))
  count = 0
  print(len(training_set_articles_data_i.keys()))
  with open(dataset_file_dir + dataset_file) as json_file:
    data = json.load(json_file)
    for d in data:
      image_id = d["image_id"]
      question_id = d["question_id"]

      if str(question_id) in training_set_articles_data_i.keys():
        count += 1
        continue

      question = d["question"]
      tokens = word_tokenize(question)
      tokens = [w.lower() for w in tokens]      
      words = [w.translate(table) for w in tokens]     
      words = [w for w in words if not w in stop_words]
      words.extend(coco_detections[str(image_id)])
      words.extend(imagenet_detections[str(image_id)])
      words.extend([p.split('/')[0] for p in places_detections[str(image_id)]])

      words = [w.lower() for w in words]

      words = list(set(words))
      words = [w for w in words if w]
      queries = list(itertools.combinations(words,2))
      articles = [wikipedia_search(q) for q in queries]
      articles = [a for a in articles if a]
      training_set_articles_data_i[question_id] = {"image_id" : image_id, "articles" : articles}
      count += 1
      if count%10 == 0:
        with open(target_dir + dataset_file,'w') as f:
          json.dump(training_set_articles_data_i,f)

      print("File : " + str(i),count, flush=True)

  with open(target_dir + dataset_file,'w') as f:
    json.dump(training_set_articles_data_i,f)


# processed_count = [170,300,170,170,170,140]

n = 6
processes = []
for i in range(n):
  pi = multiprocessing.Process(target=prepare_data, args=(i, )) 
  pi.start()
  processes.append(pi)

for i in range(n):
  processes[i].join()



