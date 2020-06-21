import json
dataset_file = "OpenEnded_mscoco_train2014_questions.json"
target_dir = "train2014_questions_split/"
data = {}

with open(dataset_file) as json_file:
	data = json.load(json_file)

def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

n = 5
qs = data["questions"]
print(len(qs))
qs_split = partition(qs,n)

for i in range(n):
	with open(target_dir + str(i) + ".json", 'w') as f:
		json.dump(qs_split[i],f)
