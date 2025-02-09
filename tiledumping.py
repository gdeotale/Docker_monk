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
import cv2
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator, PatchConfiguration
from wholeslidedata.annotation.labels import Label
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

from structures import Point
import mmdet
from mmdet.apis import DetInferencer

def run():
    # Read the input

    image_paths = glob(os.path.join("/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/images/", "pas-cpg/*.*")) #"images/kidney-transplant-biopsy-wsi-pas/*.*"))
    mask_paths = glob(os.path.join("/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/images/", "tissue-masks/*.*")) #"images/tissue-mask/*.*"))
    
    image_path = image_paths[0]
    mask_path = mask_paths[0]

    output_path = "/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/G_data/"

    patch_shape = (512, 512, 3)
    spacings = (0.5,)
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
    cnt = 0
    for x_batch, y_batch, info in tqdm(iterator, disable=True):
        x_batch = x_batch.squeeze(0)
        y_batch = y_batch.squeeze(0)
        image_bgr = cv2.cvtColor(x_batch[0], cv2.COLOR_RGB2BGR)
        print(np.unique(y_batch[0]))
        print(y_batch[0])
        # Create a color overlay for the annotations
        dot_overlay = np.zeros_like(image_bgr)
        
        dot_overlay[:, :, 0] = y_batch[0]
        
        # Blend the original image with the overlay
        alpha = 0.5  # Transparency for the overlay
        superposed_image = cv2.addWeighted(image_bgr, 1.0, dot_overlay, alpha, 0)
        cv2.imwrite('superposed.jpg', superposed_image)
        cnt=cnt+1
        if cnt==5:
         break



    iterator.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
