#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 13:27:20 2018

@author: asprunt
"""

import ConfigParser
import argparse
import numpy as np
import os
import tensorflow as tf

from matplotlib import pyplot as plt
import cv2
import re

def load_frozen_detection_model( PATH_TO_CKPT ):
  detection_graph = tf.Graph()
  with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')
  return detection_graph

def tf_detect(image_np,detection_graph):
    """Tensorflow object detection on a numpy image.
    Returns boxes, scores, and classes"""
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    with tf.Session(graph=detection_graph) as sess:
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
    return (boxes, scores, classes)

def image_file_paths(path):
    """Recursively get paths to WEBP files when passed a directory """
    out = [os.path.join(root, name)
             for root, dirs, files in os.walk(path)
             for name in files
             if name.endswith(".webp") ]
    #out = 4
    return out

def draw_box( image_in , boxes , scores, classes , box_slice=slice(0,1) ):
    """Draw boxes on a numpy image from tensorflow boxes """
    # TF box format (for each row) [ymin, xmin, ymax, xmax]
    b_n = np.squeeze(boxes)[box_slice,:]
    
    # Loop over the boxes specified by box_slice
    for row in b_n:
        
        # Box boundaries can outside the image. This brings them to the
        # border if that's the case, because I'm not sure what CV2 would do if
        # asked to draw a box outside the boundary
        row[row<0]=0
        row[row>1]=1
        
        # Get the image height and width
        image_height = image_in.shape[0]
        image_width  = image_in.shape[1]
        
        # Convert box boundaries from TensorFlow format to CV2 format
        top_left = ( int(image_width*row[1]) , int(image_height*row[0]) )
        bottom_right = ( int(image_width*row[3]) , int(image_height*row[2]) )
        
        # At last, draw the box (always in green, sorry)
        cv2.rectangle(image_in,top_left,bottom_right,(0,255,0),3)
    return image_in

def draw_box2( image_in , boxes , scores, classes , id_to_label , plt_classes ):
    """Draw boxes (with labels for the class and score) on a numpy image from
    TensorFlow boxes. Applies thresholds specified in PLT_CLASSES. Outputs a
    boolean indicating whether any thresholds exceeded"""
    
    # Initialize top_object
    top_object = []
    
    # Get the matplotlib color cycle  - for use when drawing boxes
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Font for writing the class labels and scores
    font = cv2.FONT_HERSHEY_SIMPLEX
    # TF box format (for each row) [ymin, xmin, ymax, xmax]
    
    # Loop over the classes in PLT_CLASSES (i.e. those for which detection 
    # thresholds have been provided)
    object_detected = False
    
    # I need a counter because I'm cycling through the color cycle
    ii=0
    for p_class in plt_classes:
        
        # Draw boxes of the selected class whose scores exceed the provided threshold
        plt_ind = np.where(np.squeeze(np.logical_and( classes==p_class[0],scores >= p_class[1])))
        if plt_ind[0].size:
            
            object_detected = True
            
            # Draw boxes for this class that exceed the threshold value
            for p_ind in np.nditer(plt_ind):
                b_n = np.squeeze(boxes)[p_ind,:]
                
                # Pull the edges of the box boundaries onto the image border if 
                # they're outside
                b_n[b_n<0]=0
                b_n[b_n>1]=1
                
                # Get the image height and width (TensorFlow results are
                # normalized and CV2 wants pixels)
                image_height = image_in.shape[0]
                image_width  = image_in.shape[1]
                
                # Get the coordiantes of the top left and bottom right corner
                # of the box. Remember that CV2 and NUMPY put (0,0) in the upper
                # left corner of the image
                top_left = ( int(image_width*b_n[1]) , int(image_height*b_n[0]) )
                bottom_right = ( int(image_width*b_n[3]) , int(image_height*b_n[2]) )
                
                # Draw the bounding box
                cv2.rectangle( image_in
                              ,top_left, bottom_right
                              ,hex_to_rgb(color_cycle[ii % len(color_cycle)])  ,3)
                
                # Write the class label at the top left of the box
                cv2.putText( image_in 
                            ,id_to_label.get(p_class[0])
                            ,(top_left[0], top_left[1]-image_height/40)
                            , font,1,hex_to_rgb(color_cycle[ii % len(color_cycle)]),2) #,cv2.LINE_AA)
                
                # Write the score in the lower right
                cv2.putText( image_in 
                            ,"{:.2f}".format(scores[0,p_ind])
                            ,(bottom_right[0]+image_width/100 , bottom_right[1])
                            , font,1,hex_to_rgb(color_cycle[ii % len(color_cycle)]),2) #,cv2.LINE_AA)
                
                top_object.append( ( scores[0,p_ind] , id_to_label.get(p_class[0])  )  )
            ii=ii+1
    return (object_detected , image_in , top_object)

def hex_to_rgb( h ):
    """Convert colors specified in HEX to RGB """
    return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2 ,4))

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
def load_tf_dict( dict_file ):
    """Load the dictionary of TensorFlow classes. I'm not sure where the format
    of this file is defined, but this seems to work"""
    
    # Initialize the dictionary
    id_to_label = {}
    with open( dict_file, 'r') as myfile:
        
        # Load the file and remove the new line characters
        data=myfile.read().replace('\n', '')
        
        # Split at the closing curly brace, which seems to delimit the different entries
        qq = data.split('}')
        for p in qq:
            # Regular expression to get the ID and name of each class
            match = re.search(r'.*id:\s(\d+)\s+display_name:\s"(.+)"', p.lstrip('item {  ') )
            
            # If there was a match (and there should have been), add a new entry
            # to the dictionary
            if match:
                id_to_label[int(match.group(1))] = match.group(2)
            elif len(p)!=0:
                print 'Warning: should have matched: ' + p
    return id_to_label

def isgrayscale( image_in ):
    """Detects images that though stored as RGB are actually grayscale"""
    out = np.max(np.subtract( image_in,np.mean( image_in , 2 )[:,:,np.newaxis] ))
    if out<3:
        out = True
    else:
        out = False
    return out

def store_results( res_file_prefix , classes , scores):
    """Store TensorFlow class and score results in separate text files. Each
    line in the text file will correspond to the results for a single image."""
    
    # If the directory for storing results doesn't exist, make it
    dir_name = os.path.dirname(res_file_prefix)
    if not os.path.isdir(dir_name):
        os.makedirs (dir_name)
    
    # Files to store the results in
    f_class_h = file(res_file_prefix + '_classes.txt', 'a')
    f_score_h = file(res_file_prefix + '_scores.txt', 'a')
    
    # Store the class labels
    np.savetxt(f_class_h, classes , newline=' ', delimiter=',')
    f_class_h.write('\n')
    
    # Store the scores
    np.savetxt(f_score_h, scores , newline=' ', delimiter=',')
    f_score_h.write('\n')
    
    # Close both files
    f_class_h.close()
    f_score_h.close()

def get_config_details( config_file_path ):
    """Load configuration information from a JSON file"""
    config = ConfigParser.ConfigParser()
    config.read(config_file_path)    
    return config

def batch_process_images( config_details , file_list ):
    """Run TensorFlow object detection on a list of files. Bounding boxes are 
    drawn on images where detection thresholds are met and images are saved to
    disk in the appropriate directory. If no objects are detected in an image,
    a symbolic link is created in the appropriate directory instead. All score
    and class information is saved for histogram plotting.
    
    Since the same frozen detection graph doesn't work on both RGB and grayscale
    images, results for the two are stored separately. At this time, the code
    does not support using a separate frozen graph on grayscale images, but it
    should.
    
    The intent here is that an existing library of images can be processed to 
    test out a detection graph. Score and class information is retained so that
    it may be used for setting the threshold values for operational use."""
    
    # Get output directories from configuraiton file
    out_dir    = config_details.get('object_detection', 'batch_detection_folder')
    gs_out_dir = config_details.get('object_detection', 'gs_sub_dir')   # grayscale image no object detected
    go_out_dir = config_details.get('object_detection', 'go_sub_dir')   # grayscale image object detected
    od_out_dir = config_details.get('object_detection', 'od_sub_dir')  # color image object detected
    no_out_dir = config_details.get('object_detection', 'no_sub_dir')  # color image no object detected
    
    # Create output directories if they don't exist
    for needed_dir in (gs_out_dir , od_out_dir , no_out_dir , go_out_dir):
        needed_dir = os.path.join(out_dir,needed_dir)
        if not os.path.isdir(needed_dir):
            os.makedirs (needed_dir)
    
    # Load the frozen detection graph and the corresponding class dictionary
    # Processing using only the RGB graph is currently hard coded    
    detection_graph = load_frozen_detection_model( 
            config_details.get('object_detection' , 'rgb_frozen_model_path' )
            )
    id_to_label = load_tf_dict( 
            config_details.get('object_detection' , 'rgb_dict_file' )
            )
    
    # Hard coded not to show results, but just to write them out to disk
    make_plots = False
    
    # (class_id , threshold)
    #plt_classes = [ (1,0.30) , (2,0.20) , (16,0.20) , (17,0.20) , (18,0.40) ]
    """ The classes of things we actually want to detect (as opposed to all
    the classes the frozen TensorFlow graph can detect) are indicated in the 
    configuration file along with the detection thresholds. This bit of code
    reverses the dictionary to go from class names (in the configuraiton file)
    to class IDs and then creates a list of tuples pairing each class ID with 
    its detection threshold."""
    plt_classes = []
    for cls, thresh in zip( config_details.get('object_detection','det_classes').split(',') , 
        map(float,config_details.get('object_detection','det_thresholds').split(',')) ):
            key = next(key for key, value in id_to_label.items() if value == cls.strip())
            plt_classes.append( (key , thresh) )
    
    # Loop over each image
    with detection_graph.as_default():
        for image_path in file_list:
            
            # Get the file name and extension of the image
            fname = os.path.splitext(os.path.basename(image_path))[0]
            f_ext = os.path.splitext(os.path.basename(image_path))[1]
            
            # Load the image and convert from BGR (CV2) to RGB (numpy and TensorFlow)
            image_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            
            # Is the image really grayscale?
            is_gs = isgrayscale( image_np )
            
            # Run TensorFlow object detection
            (boxes , scores, classes) = tf_detect(image_np,detection_graph)
            
            # Get the appropriate file prefix (for RGB or grayscale)
            if is_gs:
                res_file_prefix = config_details.get('object_detection','gs_hist_prefix')
            else:
                res_file_prefix = config_details.get('object_detection','rgb_hist_prefix')    
            
            # Store the score and class results
            store_results( os.path.join(out_dir,res_file_prefix) , classes , scores)
            
            # Use box drawing function to apply detection threshold criteria
            # and then to draw boxes (if appropriate)
            (object_detected , image_np ) = draw_box2( image_np 
                    , boxes, scores , classes, id_to_label, plt_classes)
            
            # If an object was detected, save a copy of it (with bounding boxes)
            # in the appropriate folder
            if object_detected:
                if make_plots:
                    plt.figure( figsize=(10,7) )
                    plt.imshow( image_np )
                    plt.show()
                
                # Create the new file by appending _sd to the name, also put it
                # into the correct directory depending on RGB or grayscale
                fname_out = fname + '_sd' + f_ext
                if is_gs:
                    out_path = os.path.join(out_dir,go_out_dir,fname_out)
                else:
                    out_path = os.path.join(out_dir,od_out_dir,fname_out)    
                
                # Write the annotated image to disk
                cv2.imwrite( out_path
                            ,cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) )
            else:
                if make_plots:
                    plt.figure( figsize=(10,7) )
                    plt.imshow( cv2.putText( image_np 
                                ,'BELOW_THRESH'
                                ,( image_np.shape[1]/2 , image_np.shape[0]/2)
                                , cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0) ,2 ) )
                    plt.show()
                
                # Create symbolic link path based on input file name an whether 
                # it's RGB or grayscale.
                fname_out = fname + f_ext
                if is_gs:
                    out_path = os.path.join(out_dir,gs_out_dir,fname_out)
                else:
                    out_path = os.path.join(out_dir,no_out_dir,fname_out)   
                
                # Create the desired symbolic link. Using try/except because
                # there may be duplicate file names. Not worth resolving this,
                # so just let the link go to the first file with the offending
                # name.
                try:
                    os.symlink(image_path,out_path)
                except:
                    print image_path

def plot_histograms( config_details , classes ):
    """Loads archived object detection data and plots histograms for the
    specified class names"""
    
    # Load the class dictionary
    id_to_label = load_tf_dict( config_details.get('object_detection' , 'rgb_dict_file' ) )
    
    # Load the stored detection results
    cls_data = np.loadtxt( os.path.join(
                            config_details.get('object_detection','batch_detection_folder'),
                            config_details.get('object_detection','rgb_hist_prefix')
                            +'_classes.txt') ,delimiter=',' )
    scr_data = np.loadtxt( os.path.join(
                            config_details.get('object_detection','batch_detection_folder'),
                            config_details.get('object_detection','rgb_hist_prefix')
                            +'_scores.txt') ,delimiter=',' )
    
    # Loop over the specified classes plotting histograms of score results.
    for cls in classes:
        
        # This reverses the class dictionary to go from class names (in the 
        # config file) to class IDs (which is how the results are stored)
        key = next(key for key, value in id_to_label.items() if value == cls.strip())
        
        # if there are any results for the class, plot them with an appropriate
        # title and axis labels.
        if np.any(cls_data.ravel()==key):
            plt.figure( figsize=(10,7) )
            plt.title('Results for class '+cls.upper())
            plt.xlabel('Score')
            plt.ylabel('Occurances')
            n, bins, patches = plt.hist( scr_data[cls_data==key] , 20 ,log=True)
            plt.show()
    return

def main():
    """Implement a command line interface"""
    parser = argparse.ArgumentParser(description='TensorFlow based object detection in images')
    parser.add_argument('mode', type=str, choices=['process', 'histogram'], help='Process specified files or produce histogram of results')
    parser.add_argument('inputs', type=str, nargs='+',
                    help='Files to process or class names (not IDs) to plot histograms for')
    parser.add_argument('-c', '--config_file', metavar='', type=str, required=True,
                        help='configuration file')
    args = parser.parse_args()
    
    # Load the configuration file
    config_details = get_config_details( args.config_file )
    
    # Process images or create histograms
    if args.mode.upper() == 'PROCESS':
        full_paths = [os.path.join(os.getcwd(), path) for path in args.inputs]
        # Build the list of files from the inputs, which may include paths to
        # files as well as directories that will be recursively walked to get
        # image files
        file_list = []
        for fname in full_paths:
            if os.path.isdir( fname ):
                file_list.extend( image_file_paths(fname) )
            else:
                file_list.append( fname )
        batch_process_images( config_details , file_list )
    elif args.mode.upper() == 'HISTOGRAM':
        plot_histograms( config_details , args.inputs )

if __name__ == '__main__':
  main()