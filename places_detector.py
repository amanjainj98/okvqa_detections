import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import numpy as np
import os
from PIL import Image
import json


arch = 'resnet18'

model_file = '%s_places365.pth.tar' % arch

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

file_name = 'categories_places365.txt'

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)


def get_places(img_file):
	img = Image.open(img_file).convert("RGB")

	input_img = V(centre_crop(img).unsqueeze(0))


	logit = model.forward(input_img)
	h_x = F.softmax(logit, 1).data.squeeze()
	probs, idx = h_x.sort(0, True)
	return [classes[idx[0]],classes[idx[1]]]


dataset_file = "OpenEnded_mscoco_train2014_questions.json"
coco_image_dir = "/mnt/data/aman/block/block.bootstrap.pytorch/data/vqa/coco/raw/train2014"
count = 0
detected_places = {}
with open(dataset_file) as json_file:
    data = json.load(json_file)
    for i in data["questions"]:
      image_id = i["image_id"]
      n = len(str(image_id))
      image_file = "COCO_train2014_" + "0"*(12-n) + str(image_id) + ".jpg"
      img_file_path = os.path.join(coco_image_dir, image_file)
      v = get_places(img_file_path)
      detected_places[image_id] = v
      print(count)
      count+=1


with open("detected_places.json",'w') as f:
	json.dump(detected_places,f)

