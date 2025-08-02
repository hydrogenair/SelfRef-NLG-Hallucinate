import os
import json
import jsonlines
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor

files = list(os.listdir(""))
print(len(files))
all_num = 0

tasks = []
with open('../selected_tasks.txt') as f:
    for l in f.readlines():
        if "#" not in l:
            tasks.append(l.strip())
print(tasks)           
# no_tasks = set()
for i, file in tqdm(enumerate(files)):
    with open(f"") as f:
        if not file.endswith("json"):
            continue
        # try:
        data = json.load(f)
        if any([c not in tasks for c in data['Categories']]):
            continue
        assert len(data['Categories']) == 1
        category = "_".join(data['Categories'][0].strip().split())
        if file in ["task288_gigaword_summarization.json"]:
            data['Definition'] = [data['Definition'][1]]
        if len(data['Definition']) > 1:
            print(file, data['Definition'])
        assert len(data['Definition']) == 1
        instruction = data['Definition'][0]
        instances = data['Instances']
        X_train, X_test_val, y_train, y_test_val = train_test_split(instances, list(range(len(instances))), test_size=0.2, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
        # except:
        #     print("wrong", file)
        file = file[:-5]
        os.makedirs(f"/{category}/{file}", exist_ok=True)
        with jsonlines.open(f"../processed/{category}/{file}/train.jsonl", 'w') as writer:
            for line in X_train:
                line2 = {
                    "id": line["id"], 
                    "question": f"{instruction}\n{line['input']}",
                    "input": line['input'],
                    "answer": line["output"]
                }
                writer.write(line2)
                
        with jsonlines.open(f"../processed/{category}/{file}/test.jsonl", 'w') as writer:
            for line in X_test:
                line2 = {
                    "id": line["id"], 
                    "question": f"{instruction}\n{line['input']}",
                    "input": line['input'],
                    "answer": line["output"]
                }
                writer.write(line2)

        with jsonlines.open(f"../processed/{category}/{file}/val.jsonl", 'w') as writer:
            for line in X_val:
                line2 = {
                    "id": line["id"], 
                    "question": f"{instruction}\n{line['input']}",
                    "input": line['input'],
                    "answer": line["output"]
                }
                writer.write(line2)