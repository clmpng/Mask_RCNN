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
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist.'

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
    NUM_CLASSES = 1 + 2  # background + 1 (hen)

    # Define image size
    IMAGE_MIN_DIM = None 
    IMAGE_MAX_DIM = 896 # must be dividable by 2 at least 6 times 

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 50 # 500

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 10 # 5
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 200
    RPN_TRAIN_ANCHORS_PER_IMAGE =  256
    MAX_GT_INSTANCES = 20 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000
    
config = TrainingConfig()

class InferenceConfig(TrainingConfig):
      DETECTION_MIN_CONFIDENCE = 0.98
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
                
                #print("adding:")
                #print(image_id)
                #print(image_path)
                # Add the image using the base method from utils.Dataset (if file exists)
                if os.path.isfile(image_path):
                 if not hasCrowd:  # to avoid that images of object with "multiple parts" are added
                  for annotation in image_annotations:
                     id = (annotation['category_id'])+1
                     print(id)
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
        scores = []
        
        for annotation in annotations:
            class_id = annotation['category_id']+1
            if class_id ==1: # if annotation is hen, then the score is considered, otherwise it is -1 
               attributes = [element for element in  annotation['extra']['attributes'] if element.isdigit()]#
               if len(attributes) > 0:
                  score = attributes[0]
               else: score = -1 
            else: 
               score = -1 
               print("Missing Score at image", image_id)
            #print("SCORE:")
            #print(score) 
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)
                scores.append(score)


        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        scores = np.array(scores, dtype=np.float32)
        print(scores)
        print(class_ids)
        return mask, class_ids, scores

dataset_train = CocoLikeDataset()
if(sys.argv[1] == "train"):
   dataset_train.load_data('/media/christian/SamsungSSD/ZED/datasets/1500_images_with_damages/output.json', '/media/christian/SamsungSSD/ZED/datasets/1500_images_with_damages/train/')
   dataset_train.load_data('/media/christian/SamsungSSD/ZED/datasets/1500_images_with_damages/output.json', '/media/christian/SamsungSSD/ZED/datasets/1500_images_with_damages/without_background/train/')
   dataset_train.prepare()
   print(dataset_train.num_classes)

dataset_val = CocoLikeDataset()
dataset_val.load_data('/media/christian/SamsungSSD/ZED/datasets/1500_images_with_damages/output.json', '/media/christian/SamsungSSD/ZED/datasets/1500_images_with_damages/val/')
dataset_val.prepare()
	
#### Training ####

if(sys.argv[1] == "train"):
   config.display()
   # Create model in training mode
   model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
   print(model.keras_model.summary())
   # Which weights to start with?
   init_with = "coco"  # imagenet, coco, or last

   if init_with == "imagenet":
       model.load_weights(model.get_imagenet_weights(), by_name=True,
                        exclude=["conv1", "mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask", "mrcnn_score_fc", "mrcnn_score"])
   elif init_with == "coco":
       # Load weights trained on MS COCO, but skip layers that
       # are different due to the different number of classes
       # See README for instructions to download the COCO weights
       model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["conv1", "mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask", "mrcnn_score_fc", "mrcnn_score"])
   elif init_with == "last":
       # Load weights from previous training 
       #model.load_weights(model.find_last(), by_name=True)
       weights_path = os.path.join(ROOT_DIR, "/media/christian/SamsungSSD/tensorflow_logs/...")  
       model.load_weights(weights_path, by_name=True,
                       exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask", "mrcnn_score_fc", "mrcnn_score"])

   # define augmentations: left-right flip, crop, and brightness multiplication
   augmentation = iaa.SomeOf((0,3), [
                    iaa.Fliplr(0.5),
                    iaa.Affine(scale=(0.5, 1.5)),
                    iaa.Multiply((0.6, 1.5)) 
                ])
   
   # Train the head branches and the first conv layer (because of additional depth channel)
   #start_train = time.time()
   #model.train(dataset_train, dataset_val, 
   #         learning_rate=config.LEARNING_RATE, 
   #         epochs=6, 
   #         layers='heads')
   #end_train = time.time()
   #minutes = round((end_train - start_train) / 60, 2)
   #print(f'Training took {minutes} minutes')
   
   # Fine-tune all layers
   start_train = time.time()
   model.train(dataset_train, dataset_val, 
               learning_rate=config.LEARNING_RATE,
               epochs=2000, 
               layers="all",
               augmentation=augmentation,
               use_depth=False)
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
       squared_errors_total = [[]]*len(self.dataset.class_info)  # squared score error for each class
       gt_scores_total = [[]]*len(self.dataset.class_info)  # all gt_scores for each class
       pred_scores_total = [[]]*len(self.dataset.class_info)  # all predicted scores for each class
       true_score_classes = [0]*len(self.dataset.class_info)  # number of correct score classes (1-3) for each class
       for index,image_id in enumerate(self.dataset.image_ids):
         print('\n')
         print("Processing image:id ",image_id) 
         # load image, bounding boxes and masks for the image id
         image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_score = modellib.load_image_gt(self.dataset, self.cfg,image_id,use_depth=False)
         
         sample = np.expand_dims(image, 0)
         # print(len(image))
         # make prediction
         yhat = self.model.detect(sample, verbose=1)
         # extract results for first sample
         r = yhat[0]
         # calculate statistics, including AP for each class
         for items in self.dataset.class_info:
          if items["id"] >= 1: # do not perform calculations for background class 
            class_name = items["name"]
            class_id = items["id"]
            print("Class_id: ", r["class_ids"])
            print("Props: ", r["class_scores"])
            print("Feather_Scores: ", r["scores"])
            gt_match, pred_match, overlaps = compute_matches(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["class_scores"], r['masks'], class_id)
            AP, precisions, recalls, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["class_scores"], r['masks'], class_id)

            # Calculate Score Error
            pred_match_indices = pred_match[(pred_match>-1)].astype(int) # For each correctly predicted box it has the index of the matched gt box. 
            gt_match_indices = gt_match[(gt_match>-1)].astype(int) # for each correctly matched gt box it has the index of the matched predicted box   
            print("gt_score:", gt_score)
            gt_match_scores = gt_score[pred_match_indices] # gt scores of each correctly predicted box 
            print("gt_match_scores: ", gt_match_scores)
            print("pred_scores: ", r["scores"])
            positive_score_indices = np.where(gt_match_scores>-1) # indices of positive gt scores 
            gt_positive_scores = gt_match_scores[positive_score_indices] 
            pred_positive_scores = r["scores"][positive_score_indices]
            print("positive_gt_scores: ", gt_positive_scores) 
            print("positive_pred_scores: ", pred_positive_scores) 
            gt_scores_total[class_id] = np.concatenate((gt_scores_total[class_id], gt_positive_scores))
            pred_scores_total[class_id] = np.concatenate((pred_scores_total[class_id], pred_positive_scores))
            error = np.absolute(np.subtract(gt_positive_scores,pred_positive_scores))
            squared_error = np.square(error)
            squared_error_image = np.sum(squared_error)
            print("MSE image: ", squared_error_image/len(positive_score_indices))
            squared_errors_total[class_id] = np.concatenate((squared_errors_total[class_id], squared_error))
            print("Current MSE: ", np.mean(squared_errors_total[class_id]))
            #categorize scores in 3 classes and evaluate accuracy:
            correct_score_class_count = len(np.where((np.round(gt_positive_scores)-np.round(pred_positive_scores))==0)[0])
            print("Correct Score Classes:", correct_score_class_count)
            true_score_classes[class_id] = true_score_classes[class_id] + correct_score_class_count
            if len(squared_errors_total[class_id]) > 0:
               score_accuracy = true_score_classes[class_id]/len(squared_errors_total[class_id])
            else:
               score_accuracy = 0 
            print("Current Score Accuracy: ", score_accuracy)
            # calculate and sum up true positives and detections
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
            #print("(Predicted) Scores:" , gt_scores_total, pred_scores_total)
            precisions_dict[image_id] = np.mean(precisions)
            recall_dict[image_id] = np.mean(recalls)
            # store
            APs.append(AP)
            
            #title = str(image_id) + 'no_mask'
            visualize.display_instances(image[..., :3], r['rois'], r['masks'], r['class_ids'],
                                   dataset_val.class_names, r['class_scores'], r['scores'], figsize=(8,8), show_mask=False, savepath=image_id) # added scores $
            #visualize.display_differences(image[..., :3], gt_bbox, gt_class_id, gt_mask, gt_score, r['rois'], r['class_ids'], r['class_scores'],
            #                       r['scores'], r['masks'], dataset_val.class_names,class_id, show_mask=False, savepath=image_id)

            #Analyze activations 
            import tensorflow as tf
            activations = self.model.run_graph([image], [
            ("input_image",        tf.identity(self.model.keras_model.get_layer("input_image").output)),
            ("conv1",          self.model.keras_model.get_layer("conv1").output),
            ("res2c_out",          self.model.keras_model.get_layer("res2c_out").output),
            ("res3c_out",          self.model.keras_model.get_layer("res3c_out").output),
            ("fpn_p4",          self.model.keras_model.get_layer("fpn_p4").output),
            ("rpn_bbox",           self.model.keras_model.get_layer("rpn_bbox").output),
            ("roi",                self.model.keras_model.get_layer("ROI").output),
            ])

            conv_sum = activations["conv1"][0,:,:,:].mean(axis=2)
            conv_sum=np.expand_dims(conv_sum, axis=2)
            res2c_sum = activations["res2c_out"][0,:,:,:].mean(axis=2)
            res2c_sum=np.expand_dims(res2c_sum, axis=2)
            print(activations["conv1"][0,:,:,:].shape)
            print(conv_sum.shape)
            #visualize.display_images(np.transpose(activations["input_image"][0,:,:,:4], [2, 0, 1]),titles=[image_id,2,3,4,5,6])
            #visualize.display_images(np.transpose(activations["conv1"][0,:,:,:], [2, 0, 1]),titles=["Conv1", image_id, 3, 4,5,6,7,8,9,10])
            #visualize.display_images(np.transpose(conv_sum, [2, 0, 1]),titles=["Conv1_Sum", image_id])
            #visualize.display_images(np.transpose(res2c_sum, [2, 0, 1]),titles=["Res2C_Sum", image_id])
            #visualize.display_images(np.transpose(activations["res3c_out"][0,:,:,:5], [2, 0, 1]),titles=["RES3C", image_id,3 ,4,5,6,7,8,9,10])

       # calculate the mean AP, precision, recall, f1, MSE  across all images
       mAP = np.mean(APs)
       for items in self.dataset.class_info:
            class_name = items["name"]
            if class_name != "Hen":
               continue
            class_id = items["id"]
            precision = true_positives_total[class_id]/detections_total[class_id]
            recall = true_positives_total[class_id]/gt_instances_total[class_id]
            MSE = np.mean(squared_errors_total[class_id]) 
            print(class_name + " :")
            print("mAp: ", mAP)
            print("precision: ", precision)
            print("recall: ", recall) 
            print("MSE: ", MSE)

       np.savetxt(f, gt_scores_total[class_id], delimiter=',')
       np.savetxt(f, pred_scores_total[class_id], delimiter=',')
       f.write('Precision: {0} Recall: {1} MSE: {2}\n'.format(precision, recall, MSE))
       return 0 

   #Create model in inference mode 
   model_inference = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
   print(model_inference.keras_model.summary())
   # Get path to saved weights
   # Either set a specific path or find last trained weights
   model_path = os.path.join(ROOT_DIR, "/media/christian/SamsungSSD/tensorflow_logs/score_training_1477_epochs_with_feather_damages/mask_rcnn_feather__damage_1400.h5")
   # model_path = model.find_last()

   # Load trained weights (fill in path to trained weights here)
   print("Loading weights from ", model_path)
   model_inference.load_weights(model_path, by_name=True)
   
   f = open("evaluation_log.txt", "a")
   f.write('Evaluation model with weights {}\n'.format(model_path))
   eval = EvalImage(dataset_val,model_inference,inference_config)
   eval.evaluate_model()
   f.close()
