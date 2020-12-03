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
   print("Please specify arguments 'train' or 'eval'")
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

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
	
class TrainingConfig(Config):
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
    NUM_CLASSES = 1 + 2  # background + 2 (feather damage + no feather damage)

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
    
config = TrainingConfig()

class InferenceConfig(TrainingConfig):
      DETECTION_MIN_CONFIDENCE = 0.85
      USE_MINI_MASK = False 

inference_config = InferenceConfig()
class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        min_file_size = 10000000
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id'] 
            class_name = category['name'] 
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
         
        classes = []     # to have a list which just contains the class names and the number of instances for each 
        for element in self.class_info:
           class_entry = {
                         'name' : element['name'],
                         'count' : 0
                         }
           classes.append(class_entry)
 
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    #image_file_name = format(image['id'], '08d')+"_"+image['file_name']
                    image_file_name = image['file_name'] 
                    print(image_file_name)
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                #find_file = glob.glob(images_dir + "/**/" + format(image['id'], '08d')+"*", recursive = True) # search in all subdirectories 
                find_file = glob.glob(images_dir + "/**/" + image_file_name, recursive = True) # search in all subdire$
                print(find_file)
                if len(find_file) >0:
                   image_path = os.path.abspath(os.path.join(find_file[0]))               
                else:
                   image_path = "not_found"
                image_annotations = annotations[image_id]
                hasCrowd = False # Quick fix to avoid problems with "count" annotations. If more than 1segmentation/object -> skip image
                for element in image_annotations:
                  if element['iscrowd']:
                    hasCrowd = True 
                
                # Add the image using the base method from utils.Dataset (if file exists)
                if os.path.isfile(image_path):
                 if not hasCrowd:  # to avoid that images of object with "multiple parts" are added
                  for annotation in image_annotations:
                     id = (annotation['category_id'])
                     classes[id]['count'] +=1
                  self.add_image(
                      source=source_name,
                      image_id=image_id,
                      path=image_path,
                      width=image_width,
                      height=image_height,
                      annotations=image_annotations)

        print("...........Dataset-Info:.......................\n")
        for i in range(len(classes)):
           print ("Class: {}, Instances: {}".format(classes[i]['name'], classes[i]['count']))
 
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids

dataset_train = CocoLikeDataset()
dataset_train.load_data('/media/christian/SamsungSSD/ZED/datasets/1200_images/attribute_annotations.json', '/media/christian/SamsungSSD/ZED/datasets/1200_images/train/')
dataset_train.prepare()
print(dataset_train.num_classes)

dataset_val = CocoLikeDataset()
dataset_val.load_data('/media/christian/SamsungSSD/ZED/datasets/1200_images/attribute_annotations.json', '/media/christian/SamsungSSD/ZED/datasets/1200_images/val/')
dataset_val.prepare()
	
#### Training ####

if(sys.argv[1] == "train"):
   config.display()
   # Create model in training mode
   model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

   # Which weights to start with?
   init_with = "coco"  # imagenet, coco, or last

   if init_with == "imagenet":
       model.load_weights(model.get_imagenet_weights(), by_name=True)
   elif init_with == "coco":
       # Load weights trained on MS COCO, but skip layers that
       # are different due to the different number of classes
       # See README for instructions to download the COCO weights
       model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["conv1", "mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
   elif init_with == "last":
       # Load the last model you trained and continue training
       model.load_weights(model.find_last(), by_name=True)

   # define augmentations (applied half of the time): left-right flip and brightness multiplication
   #augmentation = iaa.Sometimes(0.5, [
   #                 iaa.Fliplr(0.5),
   #                iaa.Multiply((0.6, 1.5))
   #             ])
   
   # Fine-tune all layers
   start_train = time.time()
   model.train(dataset_train, dataset_val, 
               learning_rate=config.LEARNING_RATE,
               epochs=300, 
               layers="all")
   end_train = time.time()
   minutes = round((end_train - start_train) / 60, 2)
   print(f'Training took {minutes} minutes')

elif(sys.argv[1] == "eval"):
   
   print("Evaluation started.. ")
   inference_config.display()
   ## Model Evaluation
   from mrcnn.utils import compute_matches, compute_ap, compute_f1 
   class EvalImage():
     def __init__(self,dataset,model,cfg):
       self.dataset = dataset
       self.model   = model
       self.cfg     = cfg

 
 
     def evaluate_model(self):
       APs = list()
       precisions_dict = {}
       recall_dict     = {}
       true_positives_total = [0]*len(self.dataset.class_info) # array with true positives for each class 
       detections_total = [0]*len(self.dataset.class_info) # detections for each class  
       gt_instances_total = [0]*len(self.dataset.class_info) # gt_instances for each class 
       for index,image_id in enumerate(self.dataset.image_ids):
         print('\n')
         print("Processing image:id ",image_id) 
         # load image, bounding boxes and masks for the image id
         image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(self.dataset, self.cfg,image_id)
         # convert pixel values (e.g. center)
         #scaled_image = modellib.mold_image(image, self.cfg)
         # convert image into one sample
         sample = np.expand_dims(image, 0)
         # print(len(image))
         # make prediction
         yhat = self.model.detect(sample, verbose=0)
         # extract results for first sample
         r = yhat[0]
         # calculate statistics, including AP for each class
         for items in self.dataset.class_info:
            class_name = items["name"]
            if class_name == "BG":
               continue
            class_id = items["id"]
            gt_match, pred_match, overlaps = compute_matches(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], class_id)
            AP, precisions, recalls, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], class_id)
            #precision, recall, f1 = compute_f1(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], class_id)
            #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
            #                       dataset_val.class_names, r['scores'], figsize=(8,8))
            true_positives = sum(i> -1 for i in pred_match)
            detections = len(pred_match)
            gt_instances = len(gt_match)
            true_positives_total[class_id] = true_positives_total[class_id] + true_positives
            detections_total[class_id] = detections_total[class_id] + detections
            gt_instances_total[class_id] = gt_instances_total[class_id] + gt_instances   
            print("---- " + class_name + " ----")
            print("gt_instances: ", gt_instances)
            print("true positives: ", true_positives)
            print("detections: ", detections) 
            print("true_positives_total: ", true_positives_total[class_id])
            print("detections_total: {}", detections_total[class_id])
            precisions_dict[image_id] = np.mean(precisions)
            recall_dict[image_id] = np.mean(recalls)
            # store
            #APs.append(AP)

       # calculate the mean AP, precision, recall, f1  across all images
       #mAP = np.mean(APs)
       for items in self.dataset.class_info:
            class_name = items["name"]
            if class_name == "BG":
               continue
            class_id = items["id"]
            precision = true_positives_total[class_id]/detections_total[class_id]
            recall = true_positives_total[class_id]/gt_instances_total[class_id] 
            print(class_name + " :")
            #print("mAp: ", mAP)
            print("precision: ", precision)
            print("recall: ", recall) 

       return 0 # mAP,precisions_dict,recall_dict

   #Create model in inference mode 
   model_inference = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

   # Get path to saved weights
   # Either set a specific path or find last trained weights
   model_path = os.path.join(ROOT_DIR, "/media/christian/SamsungSSD/tensorflow_logs/attribute_training_300_epochs/mask_rcnn_feather__damage_0294.h5")
   # model_path = model.find_last()

   # Load trained weights (fill in path to trained weights here)
   print("Loading weights from ", model_path)
   model_inference.load_weights(model_path, by_name=True)

   eval = EvalImage(dataset_val,model_inference,inference_config)
   eval.evaluate_model()
