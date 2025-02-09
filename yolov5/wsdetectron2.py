from pathlib import Path
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression


class YOLOv5Predictor:
    def __init__(self, model_path, img_size=512, conf_thresh=0.2, iou_thresh=0.5, device="cuda"):
        """
        Initialize the YOLOv5 predictor for offline inference.

        Parameters:
        - model_path: Path to the YOLOv5 model weights.
        - img_size: Input size for YOLOv5 model.
        - conf_thresh: Confidence threshold for predictions.
        - iou_thresh: IoU threshold for non-max suppression.
        - device: Compute device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Check and convert model if necessary
        self.model_path = self._handle_model_path(model_path)
        self.model = DetectMultiBackend(self.model_path, device=self.device)
        
        
    def _handle_model_path(self, model_path):
# Convert .pth to .pt if necessary
        if str(model_path).endswith(".pth"):
            print(f"Converting {model_path} to .pt format...")
            model = torch.load(model_path, map_location=self.device)  # Load the .pth model
            new_path = "/tmp/model_final.pt"  # Save in a writable directory
            torch.save(model, new_path)  # Save it as .pt
            print(f"Model converted and saved to {new_path}")
            return new_path  # Return the new path
        elif str(model_path).endswith(".pt"):
            return model_path
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

    def preprocess_image(self, img):
        """
        Preprocess the image for YOLOv5 inference.

        Parameters:
        - img: NumPy array representing the image.

        Returns:
        - img_tensor: Preprocessed image as a PyTorch tensor.
        """
        if len(img.shape) == 4 and img.shape[0] == 1:
            img = img.squeeze(0)  # Shape is (1, H, W, C)
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0  # Normalize
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        return img_tensor

    def predict_on_batch(self, x_batch):
        """
        Predict on a batch of images.

        Parameters:
        - x_batch: List of images (NumPy arrays in HWC format).

        Returns:
        - predictions: List of predictions for each image in the batch.
        """
        predictions = []

        # Ensure input batch is a list of NumPy arrays
        if isinstance(x_batch, np.ndarray):
            x_batch = [x_batch]

        for img in x_batch:
            img_tensor = self.preprocess_image(img)
            # Run inference
            preds = self.model(img_tensor)
            # Apply non-max suppression
            preds = non_max_suppression(preds, self.conf_thresh, self.iou_thresh)[0]

            # Parse predictions
            img_predictions = []
            if preds is not None:
                preds = preds.cpu().numpy()
                for pred in preds:
                    x1, y1, x2, y2, conf, class_id = pred
                    # Calculate center coordinates
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)

                    # Create prediction record
                    prediction_record = {
                        "x1": x_center,
                        "y1": y_center,
                        "label": int(class_id),
                        "confidence": float(conf),
                    }
                    img_predictions.append(prediction_record)

            predictions.append(img_predictions)

        return predictions



