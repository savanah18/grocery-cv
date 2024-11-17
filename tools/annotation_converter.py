import json
import os

def convert_annotations(coco_json_path, output_dir, isbbox=True):
    with open(coco_json_path) as f:
        data = json.load(f)

    images = {image['id']: image for image in data['images']}
    categories = {category['id']: category['name'] for category in data['categories']}
    annotations = data['annotations']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for annotation in annotations:
        image_id = annotation['image_id']
        image_info = images[image_id]
        image_width = image_info['width']
        image_height = image_info['height']

        category_id = annotation['category_id']

        if isbbox:
            # special case for class 14 and 23
            if category_id == 14 or category_id == 23:
                if not annotation['isbbox']:
                    continue

            bbox = annotation['bbox']
            old_x, old_y, old_width, old_height = bbox
            x = max(0,old_x)
            y = max(0,old_y)
            width = old_x + (old_width - x)
            height = old_y + (old_height - y)
            width = min(width, image_width - x)
            height = min(height, image_height - y)

            x_center = (x + width / 2) / image_width
            y_center = (y + height / 2) / image_height
            width /= image_width
            height /= image_height

            yolo_annotation = f"{category_id} {x_center} {y_center} {width} {height}\n"
        else:
            # special case for class 14 and 23
            if category_id == 14 or category_id == 23:
                if annotation['isbbox']:
                    continue

            segmentation = annotation['segmentation'][0]  # Assuming single polygon per object
            normalized_segmentation = []
            for i in range(0, len(segmentation), 2):
                x = segmentation[i] / image_width
                y = segmentation[i + 1] / image_height
                normalized_segmentation.append(f"{x} {y}")

            yolo_annotation = f"{category_id} " + " ".join(normalized_segmentation) + "\n"

        image_filename = image_info['file_name']
        yolo_filename = os.path.splitext(image_filename)[0] + ".txt"
        yolo_filepath = os.path.join(output_dir, yolo_filename)

        with open(yolo_filepath, 'a') as yolo_file:
            yolo_file.write(yolo_annotation)

# use arge parse to get the input and output directory
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert COCO annotations to YOLO format')
    parser.add_argument('coco_json_path', help='Path to COCO JSON file')
    parser.add_argument('output_dir', help='Directory to save YOLO annotations')
    parser.add_argument('task', help='Task to perform')
    args = parser.parse_args()

    match args.task:
        case "object_detection":
            convert_annotations(args.coco_json_path, args.output_dir, isbbox=True)
        case "segmentation":
            convert_annotations(args.coco_json_path, args.output_dir, isbbox=False)
        case _:
            convert_annotations(args.coco_json_path, args.output_dir, isbbox=True)
