import os
from datasets import load_dataset
import json
import random
from tqdm import tqdm
from PIL import PngImagePlugin
import pyarrow as pa
import pyarrow.parquet as pq
import io
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


dataset = load_dataset("jacekduszenko/weightspace-images")

def create_distribution(dataset, output_dir='data'):
    os.makedirs(output_dir, exist_ok=True)
    
    categories = list(set(dataset['train']['category_label']))
    
    for category in categories:
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    
    category_name_to_indices = {}
    category_name_to_all_indices = {} 
    active_leaf_categories = {cat: set() for cat in categories} 
    
    for idx, example in tqdm(enumerate(dataset['train']), desc="Building indices and leaf categories", total=len(dataset['train'])):
        if example['category_name'] not in category_name_to_indices:
            category_name_to_indices[example['category_name']] = []
            category_name_to_all_indices[example['category_name']] = []
        category_name_to_indices[example['category_name']].append(idx)
        category_name_to_all_indices[example['category_name']].append(idx)
        
        active_leaf_categories[example['category_label']].add(example['category_name'])
    
    for category_name in category_name_to_indices:
        random.shuffle(category_name_to_indices[category_name])
    
    active_leaf_categories = {cat: list(leaves) for cat, leaves in active_leaf_categories.items()}
    
    pbar_categories = tqdm(categories, desc="Processing categories", position=0)
    
    def image_to_bytes(image):
        """Convert PIL Image to bytes"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()
    
    for category in pbar_categories:
        pbar_categories.set_description(f"Processing category: {category}")
        
        leaf_categories = active_leaf_categories[category].copy()
        
        pbar_folders = tqdm(range(15000), desc="Creating folders", position=1, leave=False)
        
        for folder_idx in pbar_folders:
            folder_name = os.path.join(output_dir, category, f"folder_{folder_idx:05d}")
            os.makedirs(folder_name, exist_ok=True)
            
            if not active_leaf_categories[category]:
                active_leaf_categories[category] = leaf_categories.copy()
            
            leaf_category = random.choice(active_leaf_categories[category])
            
            if not category_name_to_indices[leaf_category] or len(category_name_to_indices[leaf_category]) < 4:
                active_leaf_categories[category].remove(leaf_category)
                        
                category_name_to_indices[leaf_category] = category_name_to_all_indices[leaf_category].copy()
                random.shuffle(category_name_to_indices[leaf_category])
                
                if active_leaf_categories[category]:
                    leaf_category = random.choice(active_leaf_categories[category])
            
            num_images = random.randint(4, 16)
            selected_indices = category_name_to_indices[leaf_category][:num_images]
            category_name_to_indices[leaf_category] = category_name_to_indices[leaf_category][num_images:]
            
            images_data = []
            for img_idx, dataset_idx in enumerate(selected_indices):
                image = dataset['train'][dataset_idx]['image']
                image_bytes = image_to_bytes(image)
                images_data.append({
                    'image_idx': img_idx,
                    'image_bytes': image_bytes,
                })
            
            table = pa.Table.from_pylist(images_data)
            pq.write_table(table, os.path.join(folder_name, 'images.parquet'))
            
            metadata = {
                'category_label': category,
                'category_name': leaf_category,
                'num_images': len(selected_indices),
                'leaf_id': folder_idx
            }
            
            with open(os.path.join(folder_name, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            pbar_folders.set_description(f"Creating folders for {category}")
        
        pbar_folders.close()
    
    pbar_categories.close()

create_distribution(dataset, output_dir='data/weightspace-images')

