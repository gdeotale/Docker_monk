#import onnxruntime
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''def preprocess_image(image_path, input_shape):
    """
    Preprocess the image to prepare it for YOLOv5 ONNX model inference.
    
    Args:
        image_path (str): Path to the input image.
        input_shape (tuple): Expected input shape for the model (height, width).
        
    Returns:
        np.ndarray: Preprocessed image in CHW format.
        np.ndarray: Original image (for visualization).
        float: Resize ratio used to scale bounding box predictions back to original size.
    """
    # Read the image
    original_image = image_path #cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    h, w = original_image.shape[:2]
    
    # Resize while keeping aspect ratio
    model_height, model_width = input_shape
    resize_ratio = min(model_width / w, model_height / h)
    resized_width = int(w * resize_ratio)
    resized_height = int(h * resize_ratio)
    
    resized_image = cv2.resize(original_image, (resized_width, resized_height))
    # print(resized_image.shape)
    padded_image = np.zeros((model_height, model_width, 3), dtype=np.uint8)
    padded_image[:resized_height, :resized_width] = resized_image
    # print(padded_image.shape)

    # Normalize and convert to CHW format
    preprocessed_image = padded_image / 255.0  # Normalize to [0, 1]
    preprocessed_image = preprocessed_image.transpose(2, 0, 1)  # HWC to CHW
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0).astype(np.float32)
    # print(preprocessed_image.shape)

    return preprocessed_image, original_image, resize_ratio

def postprocess_predictions(predictions, resize_ratio, image_shape, conf_threshold=0.2, iou_threshold=0.45):
    """
    Post-process model predictions to extract bounding boxes, scores, and class labels.
    
    Args:
        predictions (np.ndarray): Output predictions from the model.
        resize_ratio (float): Resize ratio used during preprocessing.
        image_shape (tuple): Shape of the original image (height, width).
        conf_threshold (float): Confidence threshold for filtering detections.
        iou_threshold (float): Intersection-over-Union threshold for NMS.
        
    Returns:
        list of dict: Processed detections (bounding boxes, scores, and class labels).
    """
    # Extract predictions: [batch, num_boxes, 85] -> [x_center, y_center, width, height, conf, class_scores...]
    boxes = predictions[0, :, :4]  # First 4 columns are bbox coordinates
    scores = predictions[0, :, 4:5] * predictions[0, :, 5:]  # Objectness score * class probabilities
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Filter by confidence threshold
    keep = confidences > conf_threshold
    boxes, confidences, class_ids = boxes[keep], confidences[keep], class_ids[keep]

    # Scale boxes back to original image size
    boxes[:, 0] -= (1 - resize_ratio) / 2  # Adjust x
    boxes[:, 1] -= (1 - resize_ratio) / 2  # Adjust y
    boxes[:, 2] -= (1 - resize_ratio) / 2  # Adjust width
    boxes[:, 3] -= (1 - resize_ratio) / 2  # Adjust height
    boxes /= resize_ratio

    # Convert boxes to [x1, y1, x2, y2] format
    boxes[:, 0:2] = boxes[:, 0:2] - boxes[:, 2:4] / 2  # x_center, y_center -> x_min, y_min
    boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]       # width, height -> x_max, y_max

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=confidences.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )
    
    results = []
    for i in indices.flatten():
        results.append({
            "bbox": boxes[i],
            "confidence": confidences[i],
            "class_id": class_ids[i]
        })

    return results
'''
def coco_to_x1y1x2y2(coco_bbox):
    x_min, y_min, x_max, y_max = coco_bbox
    if x_min<0: x_min=0
    if y_min<0: y_min=0
    if x_max>512: x_max=512
    if y_max>512: y_max=512
    # x1 = x_min
    # y1 = y_min
    # x2 = x_min + width
    # y2 = y_min + height
    return [x_min/512,y_max/512,x_max/512,y_min/512]

def x1y1x2y2_to_coco(x1y1x2y2_bbox):
    x1, y1, x2, y2 = x1y1x2y2_bbox
    return [x1*512, y2*512,x2*512,y1*512]
    
def coco_to_x1y1x2y2_batch(coco_bboxes):
    """
    Convert a list of COCO format bounding boxes to [x1, y1, x2, y2] format.

    Args:
        coco_bboxes (list of list/tuple): Each element is [x_min, y_min, width, height]

    Returns:
        list of list: Each element is [x1, y1, x2, y2]
    """
    return [coco_to_x1y1x2y2(bbox) for bbox in coco_bboxes]

def x1y1x2y2_to_coco_batch(x1y1x2y2_bboxes):
    """
    Convert a list of [x1, y1, x2, y2] format bounding boxes to COCO format.

    Args:
        x1y1x2y2_bboxes (list of list/tuple): Each element is [x1, y1, x2, y2]

    Returns:
        list of list: Each element is [x_min, y_min, width, height]
    """
    return [x1y1x2y2_to_coco(bbox) for bbox in x1y1x2y2_bboxes]

