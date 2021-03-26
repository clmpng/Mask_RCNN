import os
import sys
import json
import numpy as np
import time
import glob
from PIL import Image, ImageDraw

#import augmentation
import imgaug.augmenters as iaa

# check if arguments are given
if len(sys.argv) < 2:
   print("Please specify target_folder")
   sys.exit()

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = '../../'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

# Import mrcnn libraries
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

# Directory to save logs and trained model
MODEL_DIR = "/media/christian/SamsungSSD/tensorflow_logs/"

class InferenceConfig(Config):
  
    # Give the configuration a recognizable name
    NAME = "feather_damage_prediction"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 (hen)

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 832
    IMAGE_MAX_DIM = 832 # must be dividable by 2 at least 6 times 

    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 20 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000
    
    DETECTION_MIN_CONFIDENCE = 0.85
    USE_MINI_MASK = False 
 
    
config = InferenceConfig()

def load_image_from_path(rgb_path, depth_path):
        """Load the specified image and return a [H,W,4] Numpy array.
        """
        # Load image
        image = skimage.io.imread(path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        #load depth image
        depth = skimage.io.imread(depth_path)
        #normalize depth:
        min_depth = np.min(depth)
        max_depth = np.max(depth)
        depth = (depth-min_depth) / (max_depth - min_depth)*255
        depth_image = depth[:, :, np.newaxis]
        rgbd_image = np.concatenate((image, depth_image), axis=2)
        return rgbd_image

# define the model
#model_inference = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
# load model weights
#model_path = os.path.join(ROOT_DIR, "/media/christian/SamsungSSD/tensorflow_logs/score_training_1000_epochs_3_scores/mask_rcnn_feather__damage_0999.h5")
   # model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
#print("Loading weights from ", model_path)
#model_inference.load_weights(model_path, by_name=True)

#load images
image_folder = '/media/christian/SamsungSSD/ZED/2020_09_23/images/Futterlinie_Stufe_11_2020_09_23-16_00_06_HD2K/'
print(image_folder)

for subdir, dirs, files in os.walk(image_folder):
    for file in files:
        if file.startswith("left"):
           file_name_short = file.replace('_', '.').split('.')[0]
           depth_file = "depth" + file_name_short.replace('left', '') + ".png"
           rgb_file_path = os.path.join(subdir, file)
           depth_file_path =os.path.join(subdir, depth_file)
           print(load_image_from_path(rgb_file_path, depth_file_path))

#img = load_image_from_path()
# make prediction

#yhat = self.model.detect(sample, verbose=1)
# extract results for first sample
#r = yhat[0]
# visualize the results
#title = 'predicted'
#visualize.display_instances(image[..., :3], r['rois'], r['masks'], r['class_ids'], 
#                                   dataset_val.class_names, r['class_scores'], r['scores'], figsize=(8,8), show_mask=False, savepath=title) # added scores and save option  
