import cv2
import os

def visualize_predictions(image_dir, predictions_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over the images in the test set
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)

        # Load prediction data (YOLOv5 output saved as .txt files)
        pred_file = os.path.join(predictions_dir, img_file.replace('.jpg', '.txt'))
        if not os.path.exists(pred_file):
            continue
        
        with open(pred_file, 'r') as f:
            for line in f.readlines():
                # YOLO format: class x_center y_center width height confidence
                elements = line.strip().split()
                cls, x_center, y_center, width, height, conf = map(float, elements)
                img_h, img_w = img.shape[:2]

                # Convert YOLO format to bounding box coordinates
                x1 = int((x_center - width / 2) * img_w)
                y1 = int((y_center - height / 2) * img_h)
                x2 = int((x_center + width / 2) * img_w)
                y2 = int((y_center + height / 2) * img_h)

                # Draw bounding box and label
                label = f"Class: {int(cls)}, Conf: {conf:.2f}"
                color = (0, 255, 0)  # Green for the box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the output image with annotations
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, img)

# Directories
image_dir = 'C:\Users\shaki\Documents\VSCode\ML Based Suture Pad Analysis\ML-based-suture-pad-analysis\data\labeled_data\test'
predictions_dir = 'C:\Users\shaki\Documents\VSCode\ML Based Suture Pad Analysis\ML-based-suture-pad-analysis\yolov5\runs\detect\exp\labels'  # YOLO predictions
output_dir = 'runs/detect/exp/visualized/'

# Run the visualization
visualize_predictions(image_dir, predictions_dir, output_dir)