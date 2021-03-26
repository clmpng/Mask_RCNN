
import os
import sys
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import cv2
import argparse

# Root directory of the project
ROOT_DIR = '../../'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

# Import mrcnn libraries
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, class_ids, class_names, scores=None):
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
    N = boxes.shape[0]

    colors = colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]

    for i, c in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        
        y1, x1, y2, x2 = boxes[i]
        label = class_names[class_ids[i]]
        score = scores[i] if scores is not None else None
        caption = "{} {:.3f}".format(label, score) if score else label

        # Mask
        mask = masks[:, :, i]
        image = apply_mask(image, mask, c)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), c, 2)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, c, 2)
    return image

class InferenceConfig(Config):
    """Configuration for training on the cigarette butts dataset.
    Derives from the base Config class and overrides values specific
    to the cigarette butts dataset.
    """
    # Give the configuration a recognizable name
    NAME = "feather__damage"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 (hen)

    # All of our training images are 512x512
    IMAGE_MIN_DIM = None
    IMAGE_MAX_DIM = 512 # must be dividable by 2 at least 6 times 

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 50 # 500

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 10 # 5
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 20 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000
    USE_MINI_MASK = False 
      
inference_config = InferenceConfig()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MaskRCNN Video Object Detection/Instance Segmentation')
    parser.add_argument('-v', '--video_path', type=str, default='', help='Path to video. If None camera will be used')
    parser.add_argument('-sp', '--save_path', type=str, default='', help= 'Path to save the output. If None output won\'t be saved')
    parser.add_argument('-s', '--show', default=True, action="store_false", help='Show output')
    args = parser.parse_args()

    # Directory to save logs and trained model
    MODEL_DIR = "/media/christian/SamsungSSD/tensorflow_logs/"

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    class_names = ['hen']
    
    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

    # Load weights
    model_path = os.path.join(ROOT_DIR, "/media/christian/SamsungSSD/tensorflow_logs/score_training_700_epochs_3_scores_extra_fc_layer/mask_rcnn_feather__damage_0699.h5")
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    if args.video_path != '':
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)

    if args.save_path:
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
    while cap.isOpened():
        ret, image = cap.read()
        results = model.detect([image], verbose=1)
        r = results[0]
        image = display_instances(image[..., :3], r['rois'], r['masks'], r['class_ids'], 
                                   dataset_val.class_names, r['class_scores'], r['scores'])
        if args.show:
            cv2.imshow('MaskRCNN Object Detection/Instance Segmentation', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if args.save_path:
            out.write(image)
    cap.release()
    if args.save_path:
        out.release()
    cv2.destroyAllWindows()
