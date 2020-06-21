from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
import json
import os

resnet = models.resnet152(pretrained=True)
layer = resnet._modules.get('avgpool')

transform = transforms.Compose([            
	transforms.Scale((224, 224)),                
	transforms.ToTensor(),                     
	transforms.Normalize(                      
	mean=[0.485, 0.456, 0.406],                
	std=[0.229, 0.224, 0.225]                  
 )])


def save_features(image_file,img_file_path,target_dir):
	img = Image.open(img_file_path).convert("RGB")
	img_t = transform(img)
	img_t = torch.unsqueeze(img_t, 0)

	my_embedding = torch.zeros(1,2048,1,1)
	def copy_data(m, i, o):
		my_embedding.copy_(o.data)

	h = layer.register_forward_hook(copy_data)
	resnet(img_t)
	h.remove()
	# print(my_embedding[0, :, 0, 0])
	torch.save(my_embedding[0, :, 0, 0],target_dir+image_file+".pth")

dataset_file = "OpenEnded_mscoco_train2014_questions.json"
coco_image_dir = "/mnt/data/aman/block/block.bootstrap.pytorch/data/vqa/coco/raw/train2014"
target_dir = "/mnt/data/aman/ArticleNet/extract_img_features/"
count = 0
with open(dataset_file) as json_file:
    data = json.load(json_file)
    for i in data["questions"]:
      image_id = i["image_id"]
      n = len(str(image_id))
      image_file = "COCO_train2014_" + "0"*(12-n) + str(image_id) + ".jpg"
      img_file_path = os.path.join(coco_image_dir, image_file)
      if os.path.exists(target_dir+image_file+".pth"):
        continue
      try:
      	save_features(image_file,img_file_path,target_dir)
      except:
      	print(image_file)
      print(count)
      count += 1


