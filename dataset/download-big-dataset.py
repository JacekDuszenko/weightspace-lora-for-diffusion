import os
from datasets import load_dataset
import requests
import json

response = requests.get('https://gist.githubusercontent.com/mbostock/535395e279f5b83d732ea6e36932a9eb/raw/62863328f4841fce9452bc7e2f7b5e02b40f8f3d/mobilenet.json')
hierarchy = json.loads(response.text)

def get_dog_node(node, target_id='n02084071'):
    if 'id' in node and node['id'] == target_id:
        return node
    if 'children' in node:
        for child in node['children']:
            result = get_dog_node(child, target_id)
            if result:
                return result
    return None

def get_leaf_indices(node):
    if 'children' not in node:
        return [node['index']] if 'index' in node else []
    indices = []
    for child in node['children']:
        indices.extend(get_leaf_indices(child))
    return indices

dog_node = get_dog_node(hierarchy)
if dog_node:
    dog_indices = get_leaf_indices(dog_node)
    print(f"Found {len(dog_indices)} dog breed indices under n02084071: {sorted(dog_indices)}")
else:
    print("Dog node with ID n02084071 not found")
    
def get_cat_node(node, target_id='n02121808'):
    if 'id' in node and node['id'] == target_id:
        return node
    if 'children' in node:
        for child in node['children']:
            result = get_cat_node(child, target_id)
            if result:
                return result
    return None

cat_node = get_cat_node(hierarchy)
if cat_node:
    cat_indices = get_leaf_indices(cat_node)
    print(f"Found {len(cat_indices)} cat breed indices under n02121808: {sorted(cat_indices)}")
else:
    print("Cat node with ID n02121808 not found")


def get_bird_node(node, target_id='n01503061'):
    if 'id' in node and node['id'] == target_id:
        return node
    if 'children' in node:
        for child in node['children']:
            result = get_bird_node(child, target_id)
            if result:
                return result
    return None

bird_node = get_bird_node(hierarchy)
if bird_node:
    bird_indices = get_leaf_indices(bird_node)
    print(f"Found {len(bird_indices)} bird breed indices under n01503061: {sorted(bird_indices)}")
else:
    print("Bird node with ID n01503061 not found")


def get_fish_node(node, target_id='n02512053'):
    if 'id' in node and node['id'] == target_id:
        return node
    if 'children' in node:
        for child in node['children']:
            result = get_fish_node(child, target_id)
            if result:
                return result
    return None

fish_node = get_fish_node(hierarchy)
if fish_node:
    fish_indices = get_leaf_indices(fish_node)
    print(f"Found {len(fish_indices)} fish breed indices under n02512053: {sorted(fish_indices)}")
else:
    print("Fish node with ID n02512053 not found")

def get_snake_node(node, target_id='n01726692'):
    if 'id' in node and node['id'] == target_id:
        return node
    if 'children' in node:
        for child in node['children']:
            result = get_snake_node(child, target_id)
            if result:
                return result
    return None

snake_node = get_snake_node(hierarchy)
if snake_node:
    snake_indices = get_leaf_indices(snake_node)
    print(f"Found {len(snake_indices)} snake breed indices under n01726692: {sorted(snake_indices)}")
else:
    print("Snake node with ID n01726692 not found")


def get_car_node(node, target_id='n02958343'):
    if 'id' in node and node['id'] == target_id:
        return node
    if 'children' in node:
        for child in node['children']:
            result = get_car_node(child, target_id)
            if result:
                return result
    return None

car_node = get_car_node(hierarchy)
if car_node:
    car_indices = get_leaf_indices(car_node)
    print(f"Found {len(car_indices)} car breed indices under n02958343: {sorted(car_indices)}")
else:
    print("Car node with ID n02958343 not found")


def get_aircraft_node(node, target_id='n02686568'):
    if 'id' in node and node['id'] == target_id:
        return node
    if 'children' in node:
        for child in node['children']:
            result = get_aircraft_node(child, target_id)
            if result:
                return result
    return None

aircraft_node = get_aircraft_node(hierarchy)
if aircraft_node:
    aircraft_indices = get_leaf_indices(aircraft_node)
    print(f"Found {len(aircraft_indices)} aircraft breed indices under n02686568: {sorted(aircraft_indices)}")
else:
    print("Aircraft node with ID n02686568 not found")


def get_ship_node(node, target_id='n04194289'):
    if 'id' in node and node['id'] == target_id:
        return node
    if 'children' in node:
        for child in node['children']:
            result = get_ship_node(child, target_id)
            if result:
                return result
    return None

ship_node = get_ship_node(hierarchy)
if ship_node:
    ship_indices = get_leaf_indices(ship_node)
    print(f"Found {len(ship_indices)} ship breed indices under n04194289: {sorted(ship_indices)}")
else:
    print("Ship node with ID n04194289 not found")


def get_fruit_node(node, target_id='n13134947'):
    if 'id' in node and node['id'] == target_id:
        return node
    if 'children' in node:
        for child in node['children']:
            result = get_fruit_node(child, target_id)
            if result:
                return result
    return None

fruit_node = get_fruit_node(hierarchy)
if fruit_node:
    fruit_indices = get_leaf_indices(fruit_node)
    print(f"Found {len(fruit_indices)} fruit breed indices under n13134947: {sorted(fruit_indices)}")
else:
    print("Fruit node with ID n13134947 not found")


def get_vegetable_node(node, target_id='n07707451'):
    if 'id' in node and node['id'] == target_id:
        return node
    if 'children' in node:
        for child in node['children']:
            result = get_vegetable_node(child, target_id)
            if result:
                return result
    return None

vegetable_node = get_vegetable_node(hierarchy)
if vegetable_node:
    vegetable_indices = get_leaf_indices(vegetable_node)
    print(f"Found {len(vegetable_indices)} vegetable breed indices under n07707451: {sorted(vegetable_indices)}")
else:
    print("Vegetable node with ID n07707451 not found")



dataset = load_dataset("imagenet-1k", split="train", data_files='data/train_images_0.tar.gz', cache_dir=os.environ['HF_DATASETS_CACHE'])

print('downloaded original imagenet with length', len(dataset))

all_valid_indices = set().union(
    dog_indices,
    cat_indices,
    bird_indices, 
    fish_indices,
    snake_indices,
    car_indices,
    aircraft_indices,
    ship_indices,
    fruit_indices,
    vegetable_indices
)

def find_leaf_node(node, target_label):
    if 'index' in node and node['index'] == target_label:
        return node
    if 'children' in node:
        for child in node['children']:
            result = find_leaf_node(child, target_label)
            if result:
                return result
    return None

def get_category_info(label):
    if label in dog_indices:
        return 'dog', 'n02084071'
    elif label in cat_indices:
        return 'cat', 'n02121620'
    elif label in bird_indices:
        return 'bird', 'n01503061'
    elif label in fish_indices:
        return 'fish', 'n02512053'
    elif label in snake_indices:
        return 'snake', 'n01726692'
    elif label in car_indices:
        return 'car', 'n04490091'
    elif label in aircraft_indices:
        return 'airplane', 'n02690373'
    elif label in ship_indices:
        return 'ship', 'n04194289'
    elif label in fruit_indices:
        return 'fruit', 'n13134947'
    elif label in vegetable_indices:
        return 'vegetable', 'n07707451'
    return None, None

label_to_info = {}
for label in all_valid_indices:
    category_label, hierarchy_id = get_category_info(label)
    leaf_node = find_leaf_node(hierarchy, label)
    leaf_id = leaf_node['id'] if leaf_node else None
    category_name = leaf_node['name'] if leaf_node else None
    label_to_info[label] = {
        'category_label': category_label,
        'leaf_id': leaf_id,
        'hierarchy_id': hierarchy_id,
        'category_name': category_name
    }
    
print('created label to info dict')

def add_category_info(example):
    label = example['label']
    info = label_to_info[label]
    
    image = example['image']
    resized_image = image.resize((256, 256))
    example['image'] = resized_image
    
    example['category_label'] = info['category_label']
    example['leaf_id'] = info['leaf_id']
    example['hierarchy_id'] = info['hierarchy_id']
    example['category_name'] = info['category_name']
    return example


print('started filtering the dataset')

filtered_dataset = dataset.filter(
    lambda example: example['label'] in all_valid_indices
)

print('successfully filtered the dataset, started mapping the dataset')
mapped_dataset = filtered_dataset.map(add_category_info)

print('successfully mapped the dataset, saving to file')
mapped_dataset.save_to_disk("lora-weightspace-multiclass-big")