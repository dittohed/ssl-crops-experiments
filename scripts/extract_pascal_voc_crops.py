import os
from xml.etree import ElementTree as ET
from PIL import Image


def generate_crops(input_folder, output_folder, subset, min_size=64):
    # Get subset image IDs
    subset_file = os.path.join(input_folder, 'ImageSets', 'Main', subset+'.txt')
    with open(subset_file, 'r') as f:
        subset_ids = [line.strip() for line in f.readlines()]

    # Create output folders for each class
    classes = list()
    annotations_folder = os.path.join(input_folder, 'Annotations')
    for filename in os.listdir(annotations_folder):
        xml_path = os.path.join(annotations_folder, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('.//object'):
            class_name = obj.find('name').text
            classes.append(class_name)
    
    classes = set(classes)
    for class_name in classes:
        class_folder = os.path.join(output_folder, subset, class_name)
        os.makedirs(class_folder, exist_ok=True)
    
    # Iterate through annotations and (sufficiently big) save crops
    for filename in os.listdir(annotations_folder):
        img_id = os.path.splitext(filename)[0]
        if img_id not in subset_ids:
            continue

        xml_path = os.path.join(annotations_folder, filename)
        img_filename = os.path.splitext(filename)[0] + '.jpg'
        img_path = os.path.join(input_folder, 'JPEGImages', img_filename)
        img = Image.open(img_path)
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.findall('.//object'):
            class_name = obj.find('name').text
            class_folder = os.path.join(output_folder, subset, class_name)
            
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)

            if x_max-x_min < min_size or y_max-y_min < min_size:
                continue
            
            cropped_img = img.crop((x_min, y_min, x_max, y_max))

            # Save crop to the corresponding class folder
            crop_filename = f'{os.path.splitext(img_filename)[0]}_{x_min}_{y_min}_{x_max}_{y_max}.jpg'
            crop_path = os.path.join(class_folder, crop_filename)
            cropped_img.save(crop_path)


if __name__ == '__main__':
    input_folder = './data/VOC2007'
    output_folder = './data/VOC2007/Cropped'
    
    generate_crops(input_folder, output_folder, 'train')
    generate_crops(input_folder, output_folder, 'val')