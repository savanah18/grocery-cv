# best_ckp="/data/students/gerry/repos/grocery-cv/aiml/outputs/2024-11-11/08-24-34/grocery-cv/train/weights/best.pt"

from ultralytics import YOLO

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert COCO annotations to YOLO format')
    parser.add_argument('model', help='Path to model checkpoint')
    parser.add_argument('data', help='Path to dataset configuration')
    parser.add_argument('project', help='Name of the project')
    args = parser.parse_args()
    print(args.model, args.data)

    model = YOLO(args.model)  # load a custom model
    # # Validate the model
    # print(model)
    metrics = model.val(data=args.data, project=args.project)
    # print(metrics)