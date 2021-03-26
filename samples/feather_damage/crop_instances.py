import os
import sys
import json
import numpy as np
import glob
from PIL import Image, ImageDraw
import pathlib

# check if arguments are given
#if len(sys.argv) < 2:
#   print("Please specify arguments 'train' or 'eval'")
#   sys.exit()

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = '../../'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist.'

# Import mrcnn libraries
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

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
           if class_name == "Hen":
              hen_class_id = class_id 
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
 
        # Get annotations:
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
               annotations[image_id] = []
            if annotation['category_id'] == hen_class_id-1:
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
            class_id = annotation['category_id']
            attributes = [element for element in  annotation['extra']['attributes'] if element.isdigit()]#
            if len(attributes) > 0:
               score = attributes[0]
            else: 
               score = -1 
               print("Missing Score at image", image_id)
            print("SCORE:")
            print(score) 
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

        return mask, class_ids, scores
 
class ProcessDataset():
     def __init__(self,dataset):
       self.dataset = dataset
       self.cfg=Config
 
     def delete_background(self):
       #get folder path
       #create folder  "without_background"
       for index,image_id in enumerate(self.dataset.image_ids):
         print("Processing image:id ",image_id) 
         global_path = pathlib.PurePath(self.dataset.image_info[image_id]['path'])
         print(global_path)
         path_split =  global_path.parts
         image_file = path_split[-1] 
         folder = path_split[-2]  #recording folder
         parentfolder = path_split[-3] #train/val
         directory = global_path.parents[2]
         # create new directory for rgb and depth images 
         new_rgb_directory = str(directory) +'/without_background/'+ parentfolder + '/' + folder + '/'
         new_depth_directory = str(directory) +'/without_background/depth/' + parentfolder + '/' + folder + '/' 
         pathlib.Path(new_rgb_directory).mkdir(parents=True, exist_ok=True)
         pathlib.Path(new_depth_directory).mkdir(parents=True, exist_ok=True)
         # load rgbd-image, bounding boxes and masks for the image id
         image = dataset.load_image(image_id)
         mask, class_ids, scores = dataset.load_mask(image_id)
         bbox = utils.extract_bboxes(mask)
         visualize.crop_instances(image[..., :3].astype('uint8'),bbox, mask, class_ids,
                                   dataset.class_names, scores, savepath=new_rgb_directory + image_file)
         depth_image = image[...,3]
         depth_image = depth_image[:, :, np.newaxis]
         depth_image_file = image_file.replace("left", "depth")
         depth_image_filename = depth_image_file.split('_')
         print(len(depth_image_filename))
         print(depth_image_filename)
         if len(depth_image_filename) > 1:
            depth_image_file = depth_image_filename[0] + '.png' 
    
         visualize.crop_instances(depth_image.astype('uint16'),bbox, mask, class_ids,
                                   dataset.class_names, scores, savepath=new_depth_directory + depth_image_file)  

       return 0 

dataset = CocoLikeDataset()
dataset.load_data('/media/christian/SamsungSSD/ZED/datasets/1500_images_with_damages/coco_with_feather_damages.json', '/media/christian/SamsungSSD/ZED/datasets/1500_images_with_damages/train/')
dataset.prepare()
processed_data = ProcessDataset(dataset)
processed_data.delete_background()

