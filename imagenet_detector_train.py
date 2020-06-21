import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import json



class_idx = json.load(open("imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


resnet = models.resnet152(pretrained=True)
resnet.eval()

transform = transforms.Compose([            #[1]
 transforms.Scale((224, 224)),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])



def get_objects(img_file):
  img = Image.open(img_file).convert("RGB")
  img_t = transform(img)
  t_img = torch.unsqueeze(img_t, 0)
  out = resnet(t_img)
  _, indices = torch.sort(out, descending=True)
  classes = [idx2label[idx].replace("_"," ") for idx in indices[0][:3]]
  return classes


dataset_file = "OpenEnded_mscoco_train2014_questions.json"
coco_image_dir = "/mnt/data/aman/block/block.bootstrap.pytorch/data/vqa/coco/raw/train2014"
count = 0
detected_objects = {}
with open(dataset_file) as json_file:
    data = json.load(json_file)
    for i in data["questions"]:
      image_id = i["image_id"]
      n = len(str(image_id))
      image_file = "COCO_train2014_" + "0"*(12-n) + str(image_id) + ".jpg"
      img_file_path = os.path.join(coco_image_dir, image_file)
      v = get_objects(img_file_path)
      detected_objects[image_id] = v
      print(count)
      count+=1


with open("detected_imagenet_objects.json",'w') as f:
	json.dump(detected_objects,f)

