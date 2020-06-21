import tensorflow as tf

from distutils.version import StrictVersion
print(tf.__version__)

import os

import urllib
import tarfile

import numpy as np
import multiprocessing 


MODEL_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'
MODEL_FILE = MODEL_NAME + '.tar.gz'

if os.path.exists(MODEL_FILE) is False:
    opener = urllib.request.URLopener()
    opener.retrieve(MODEL_DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

BOX_DESCRIPTIONS_FILE = 'class-descriptions-boxable.csv'
OID_DOWNLOAD_BASE = 'https://storage.googleapis.com/openimages/2018_04/'

if os.path.exists(BOX_DESCRIPTIONS_FILE) is False:
    opener = urllib.request.URLopener()
    opener.retrieve(OID_DOWNLOAD_BASE + BOX_DESCRIPTIONS_FILE, BOX_DESCRIPTIONS_FILE)


FROZEN_GRAPH_FILE = 'frozen_inference_graph.pb'
PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_NAME, FROZEN_GRAPH_FILE)

if os.path.exists(MODEL_NAME) is False:
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        filename = os.path.basename(file.name)
        if FROZEN_GRAPH_FILE in filename:
            tar_file.extract(file, os.getcwd())

import pandas as pd

ID_KEY = 'id'
CLASS_KEY = 'class'
NAME_KEY = 'name'

df = pd.read_csv(BOX_DESCRIPTIONS_FILE, names=[ID_KEY, CLASS_KEY])
category_index = {}
for idx, row in df.iterrows():
    category_index[idx+1] = {ID_KEY: row[ID_KEY], NAME_KEY: row[CLASS_KEY]}

graph = tf.Graph()
with graph.as_default():
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')


##########################################################################################



def load_image_into_numpy_array(image):
    (width, height) = image.size
    return np.array(image.getdata()) \
        .reshape((height, width, 3)).astype(np.uint8)


def process_output(classes, scores, category_index):
    results = []

    for clazz, score in zip(classes, scores):
        if score > 0.0:
            label = category_index[clazz][NAME_KEY]
            obj_result = {"label" : label, "score" : int(score*100)}
            results.append(obj_result)
    
    return results


IMAGE_TENSOR_KEY = 'image_tensor'

DETECTION_SCORES_KEY = 'detection_scores'
DETECTION_CLASSES_KEY = 'detection_classes'

TENSOR_SUFFIX = ':0'


def run_inference(graph, image_np):
    output_tensor_dict = {
        DETECTION_SCORES_KEY: DETECTION_SCORES_KEY + TENSOR_SUFFIX,
        DETECTION_CLASSES_KEY: DETECTION_CLASSES_KEY + TENSOR_SUFFIX
    }

    with graph.as_default():
        with tf.Session() as sess:
            input_tensor = tf.get_default_graph()\
                .get_tensor_by_name(IMAGE_TENSOR_KEY + TENSOR_SUFFIX)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            input_tensor_dict = {input_tensor: image_np_expanded}
            output_dict = sess.run(output_tensor_dict,
                                   feed_dict=input_tensor_dict)

            return {
                DETECTION_SCORES_KEY: 
                    output_dict[DETECTION_SCORES_KEY][0],
                DETECTION_CLASSES_KEY: 
                    output_dict[DETECTION_CLASSES_KEY][0].astype(np.int64)
            }


import PIL.Image as Image
import json
from tqdm import tqdm

img_dir = "/mnt/data/aman/block/block.bootstrap.pytorch/data/vqa/coco/raw/val2014"
target_dir = "oid_detections_split_val/"
dataset_file_dir = "val2014_questions_split/"

def object_detector(i):
    dataset_file = str(i) + ".json"
    target_file = target_dir + str(i) + ".json"
    file_result_dict = {}
    if os.path.exists(target_file):
        file_result_dict = json.load(open(target_file))
    
    count = 0

    with open(dataset_file_dir + dataset_file) as json_file:
        data = json.load(json_file)
        for d in tqdm(data, desc="File : " + str(i)):
            image_id = d["image_id"]
            if str(image_id) in file_result_dict:
                continue

            n = len(str(image_id))
            filename = "COCO_val2014_" + "0"*(12-n) + str(image_id) + ".jpg"

            image_np = load_image_into_numpy_array(Image.open(os.path.join(img_dir,filename)).convert("RGB"))
            output_dict = run_inference(graph, image_np)
            results = process_output(output_dict[DETECTION_CLASSES_KEY],
                                     output_dict[DETECTION_SCORES_KEY],
                                     category_index)

            file_result_dict[image_id] = results
            # count += 1
            # print("File " + str(i),count)
            if count%10 == 0:
                with open(target_file,'w') as f:
                    json.dump(file_result_dict,f)

        with open(target_file,'w') as f:
            json.dump(file_result_dict,f)


n = int(input()) 
object_detector(n)
