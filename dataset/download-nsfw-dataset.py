import argparse
from datasets import load_dataset
from huggingface_hub import login
import numpy as np
import uuid
import os
import json
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--huggingface_token", type=str, required=True)
parser.add_argument("--limit", type=int, default=100)
args = parser.parse_args()

login(args.huggingface_token)
dataset = load_dataset("deepghs/nsfw_detect")
imagenet = load_dataset("benjamin-paine/imagenet-1k-256x256", split="test")
base_dir = "data/images"
metadata = {}

os.makedirs(base_dir, exist_ok=True)

categories = ["hentai", "porn"]
for category in categories:
    os.makedirs(os.path.join(base_dir, category), exist_ok=True)

labels = np.array(dataset["train"]["label"])
hentai_index = np.where(labels == 1)[0]
porn_index = np.where(labels == 3)[0]

for index in tqdm(hentai_index[: args.limit]):
    record = dataset["train"][index.item()]
    image = record["image"]
    label = record["label"]
    image = image.resize((256, 256))

    image_id = str(uuid.uuid4())
    image_path = os.path.join(base_dir, "hentai", f"{image_id}.jpg")
    image.save(image_path)

    metadata[image_id] = {"category": "hentai", "path": image_path}

for index in tqdm(porn_index[: args.limit]):
    record = dataset["train"][index.item()]
    image = record["image"]
    label = record["label"]
    image = image.resize((256, 256))

    image_id = str(uuid.uuid4())
    image_path = os.path.join(base_dir, "porn", f"{image_id}.jpg")
    image.save(image_path)

    metadata[image_id] = {"category": "porn", "path": image_path}


imagenet_dir = os.path.join(base_dir, "neutral")
os.makedirs(imagenet_dir, exist_ok=True)

for i in tqdm(range(args.limit)):
    image = imagenet[i]["image"]
    image_id = str(uuid.uuid4())
    image = image.resize((256, 256))
    image_path = os.path.join(imagenet_dir, f"{image_id}.jpg")
    image.save(image_path)

    metadata[image_id] = {"category": "imagenet", "path": image_path, "label": 'neutral'}

with open("image-metadata.json", "w") as f:
    json.dump(metadata, f)
