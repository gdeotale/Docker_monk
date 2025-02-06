"""
It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

from pathlib import Path
from glob import glob
import os
import json
from tqdm import tqdm
from utils_ensemble import *

from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator, PatchConfiguration
from wholeslidedata.annotation.labels import Label
import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = Path("/input/") 
OUTPUT_PATH = Path("/output/")
RESOURCE_PATH = Path("resources")
Model_PATH = Path("/opt/ml/model1")
import numpy as np
from structures import Point
import mmdet
from mmdet.apis import DetInferencer
from ensemble_boxes_nms import *
import warnings
from wsdetectron2 import YOLOv5Predictor
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message="__floordiv__ is deprecated")

def run():
    # Read the input

    image_paths = glob(os.path.join(INPUT_PATH, "images/kidney-transplant-biopsy-wsi-pas/*.*"))
    mask_paths = glob(os.path.join(INPUT_PATH, "images/tissue-mask/*.*"))

    image_path = image_paths[0]
    mask_path = mask_paths[0]

    output_path = OUTPUT_PATH
    json_filename_lymphocytes = "detected-lymphocytes.json"
    weight_root_dino1 = os.path.join(Model_PATH, "Dino1.pth")
    weight_root_dino2 = os.path.join(Model_PATH, "Dino2.pth")
    #weight_root_dino3 = os.path.join(Model_PATH, "Dino3.pth")
    weight_root_yolov5 = os.path.join(Model_PATH, "yolov5.pth")
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    patch_shape = (512, 512, 3)
    spacings = (0.25,)
    overlap = (0, 0)
    offset = (0, 0)
    center = False

    patch_configuration = PatchConfiguration(patch_shape=patch_shape,
                                             spacings=spacings,
                                             overlap=overlap,
                                             offset=offset,
                                             center=center)
    
    iterator = create_patch_iterator(image_path=image_path,
                                     mask_path=mask_path,
                                     patch_configuration=patch_configuration,
                                     cpus=4,
                                     backend='asap')

    inferencer_dino2 = DetInferencer('/mmdetection/configs/dino/dino_swin_monkey2.py', weight_root_dino2, 'cuda')
    inferencer_dino1 = DetInferencer('/mmdetection/configs/dino/dino_swin_monkey1.py', weight_root_dino1, 'cuda')
    inferencer_yolo = YOLOv5Predictor(weight_root_yolov5, 'cuda')

    # Save your output
    inference(
        iterator=iterator,
        predictor_dino2=inferencer_dino2,
        predictor_dino1=inferencer_dino1,
        predictor_yolo=inferencer_yolo,
        spacing=spacings[0],
        image_path=image_path,
        output_path=output_path,
        json_filename=json_filename_lymphocytes,
    )

    iterator.stop()

    location_detected_lymphocytes_all = glob(os.path.join(OUTPUT_PATH, "*.json"))
    location_detected_lymphocytes = location_detected_lymphocytes_all[0]
    print(location_detected_lymphocytes_all)
    print(location_detected_lymphocytes)
    # Secondly, read the results
    result_detected_lymphocytes = load_json_file(
        location=location_detected_lymphocytes,
    )

    return 0


def px_to_mm(px: int, spacing: float):
    return px * spacing / 1000


def to_wsd(points):
    """Convert list of coordinates into WSD points"""
    new_points = []
    for i, point in enumerate(points):
        p = Point(
            index=i,
            label=Label("lymphocyte", 1, color="blue"),
            coordinates=[point],
        )
        new_points.append(p)
    return new_points

import cv2
def inference(iterator, predictor_dino2, predictor_dino1, predictor_yolo, spacing, image_path, output_path, json_filename):
    print("predicting...")
    output_dict = {
        "name": "lymphocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    output_dict_monocytes = {
        "name": "monocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    output_dict_inflammatory_cells = {
        "name": "inflammatory-cells",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    annotations = []
    counter = 0

    spacing_min = 0.25
    ratio = spacing / spacing_min
    with WholeSlideImage(image_path) as wsi:
        spacing = wsi.get_real_spacing(spacing_min)
    
    cl = 0
    cm = 0
    ci = 0
    for x_batch, y_batch, info in tqdm(iterator, disable=True):
        x_batch = x_batch.squeeze(0)
        y_batch = y_batch.squeeze(0)
        image_bgr = cv2.cvtColor(x_batch[0], cv2.COLOR_RGB2BGR)

        #predictions = inference_detector(predictor, x_batch[0])
        predictions_dino2 = predictor_dino2(image_bgr, pred_score_thr=0.3) #[predictor(x)[0] for x in x_batch]
        predictions_dino1 = predictor_dino1(image_bgr, pred_score_thr=0.3) #[predictor(x)[0] for x in x_batch]
        predictions_yolo = predictor_yolo.predict_on_batch(x_batch)

        c = info['x']
        r = info['y']
        
        bboxes_dino2 = []
        labels_dino2 = []
        scores_dino2 = []
        bboxes = predictions_dino2['predictions'][0]['bboxes']  # (N, 4) tensor of bounding boxes
        scores = predictions_dino2['predictions'][0]['scores']  # (N,) tensor of scores (confidence)
        labels = predictions_dino2['predictions'][0]['labels']  # (N,) tensor of labels (0 or 1)
        for i,j in enumerate(scores):
                bboxes_dino2.append(bboxes[i])
                scores_dino2.append(scores[i])
                labels_dino2.append(labels[i])
            
        bboxes_dino1 = []
        labels_dino1 = []
        scores_dino1 = []
        bboxes = predictions_dino1['predictions'][0]['bboxes']  # (N, 4) tensor of bounding boxes
        scores = predictions_dino1['predictions'][0]['scores']  # (N,) tensor of scores (confidence)
        labels = predictions_dino1['predictions'][0]['labels']  # (N,) tensor of labels (0 or 1)
        for i,j in enumerate(scores):
                bboxes_dino1.append(bboxes[i])
                scores_dino1.append(scores[i])
                labels_dino1.append(labels[i])

        boxes_list = []
        scores_list = []
        labels_list = []
        try:
            boxes_list.extend((coco_to_x1y1x2y2_batch(bboxes_dino2), coco_to_x1y1x2y2_batch(bboxes_dino1)))
            scores_list.extend((scores_dino2, scores_dino1))
            labels_list.extend((labels_dino2, labels_dino1))

            weights = [1,1]
            iou_thr = 0.45
            skip_box_thr = 0.0001
            sigma = 0.1
            bboxes_ens, scores_ens, labels_ens = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
            bboxes_ens = x1y1x2y2_to_coco_batch(bboxes_ens)
            #print(len(labels_ens),len(scores_ens))
            for i in range(len(scores_ens)):
                confidence = scores_ens[i]  # Get the score/confidence of the prediction
                label = labels_ens[i]       # Get the predicted label (0 or 1)

                # Get the bounding box coordinates (in this case, (x1, y1, x2, y2))
                x1, y1, x2, y2 = bboxes_ens[i]

                # Compute the center of the bounding box
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)

                if y_batch[0][y][x] == 0:
                    continue

                if x == 512 or y == 512:
                    continue

                x = x * ratio + c  # x is in spacing= 0.5 but c is in spacing = 0.25
                y = y * ratio + r

                # Convert coordinates (for example, scale the pixel values to real-world units)
                # Assuming a placeholder function px_to_mm exists to convert pixel coordinates to millimeters
                x_mm = px_to_mm(x, spacing=0.24199951445730394)  # Use your actual spacing or scaling factor
                y_mm = px_to_mm(y, spacing=0.24199951445730394)

                # Create a prediction record for this point
                prediction_record = {
                    "name": f"Point {counter}",
                    "point": [x_mm, y_mm, 0.24199951445730394],  # Placeholder for Z-coordinate
                    "probability": confidence
                }

                # Add to label-specific groups
                if label == 0:  # Lymphocyte
                    if confidence>0.225:
                        output_dict["points"].append(prediction_record)
                        cl =cl + 1

            
                if confidence>0.22:
                    output_dict_inflammatory_cells["points"].append(prediction_record)
                    ci=ci+1
            
                # Save annotations
                annotations.append((x, y))
                counter += 1
        except:
            continue

        try:        
          for idx, prediction in enumerate(predictions_yolo):

            c = info['x']
            r = info['y']

            for detections in prediction:
                x, y, label, confidence = detections.values()
                
                if x == 512 or y == 512:
                    continue

                if y_batch[idx][y][x] == 0:
                    continue

                x = x * ratio + c  # x is in spacing= 0.5 but c is in spacing = 0.25
                y = y * ratio + r
                x_mm = px_to_mm(x, spacing=0.24199951445730394)  # Use your actual spacing or scaling factor
                y_mm = px_to_mm(y, spacing=0.24199951445730394)
                prediction_record = {
                    "name": f"Point {counter}",
                    "point": [x_mm, y_mm, 0.24199951445730394],  # Placeholder for Z-coordinate
                    "probability": confidence
                }
                
                if label == 1:  # Monocyte
                    output_dict_monocytes["points"].append(prediction_record)
                    cm = cm + 1
                  

                annotations.append((x, y))
                counter += 1
        except:
            continue        
    print(f"Predicted {ci} {cl} {cm} points")
    print("saving predictions...")

    # saving json file
    output_path_json = os.path.join(output_path, json_filename)

    write_json_file(
            location=output_path_json,
            content=output_dict
        )

    json_filename_monocytes = "detected-monocytes.json"
    # it should be replaced with correct json files
    output_path_json = os.path.join(output_path, json_filename_monocytes)
    write_json_file(
            location=output_path_json,
            content=output_dict_monocytes
        )

    json_filename_inflammatory_cells = "detected-inflammatory-cells.json"
    # it should be replaced with correct json files
    output_path_json = os.path.join(output_path, json_filename_inflammatory_cells)
    write_json_file(
            location=output_path_json,
            content=output_dict_inflammatory_cells
        )

    print("finished!")

def write_json_file(*, location, content):
    # Convert NumPy types to Python-native types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable.")

    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4, default=convert_numpy))


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
