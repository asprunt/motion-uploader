#!/usr/bin/python

import os
from datetime import datetime
from uploader import MotionUploader
import argparse
import configparser
import simple_detection as sd
import cv2
import numpy as np

def send_email(subject, body, config_file_path, files=None, send_to=None):
    import smtplib

    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.utils import COMMASPACE, formatdate

    assert isinstance(files,list) | (files is None)

    config = configparser.ConfigParser()
    config.read(config_file_path)
    gmail_user = config.get('gmail', 'user') 
    gmail_pwd = config.get('gmail', 'password')
    
    # Kind of a hacky way of implenting a default, but it should work
    if send_to==None:
        send_to = [x.strip() for x in config.get('gmail' , 'recipient' ).split(',')]

    assert isinstance(send_to, list)

    msg = MIMEMultipart(
        'alternative',
        From=gmail_user,
        To=COMMASPACE.join(send_to),
        Date=formatdate(localtime=True),
    )
    msg['Subject'] = subject
    html_head = '<!DOCTYPE html><html><head></head><body>'
    html_tail = '</body></html>'
    html_body = html_head+body+html_tail
    msg.attach(MIMEText(body,'plain')) 
    msg.attach(MIMEText(html_body,'html'))
    
    for f in files or []:
        with open(f, "rb") as fil:
            msg.attach(MIMEApplication(
                fil.read(),
                Content_Disposition='attachment; filename="%s"' % os.path.basename(f),
                Name=os.path.basename(f)
            ))

    server_ssl = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    server_ssl.ehlo() # optional, called by login()
    server_ssl.login(gmail_user, gmail_pwd)
    
    server_ssl.sendmail(gmail_user, send_to, msg.as_string() )
    server_ssl.close()


def main():
    """Implement a command line interface"""
    parser = argparse.ArgumentParser(description='Video handling script for Motion')
    parser.add_argument('mode', type=str, choices=['send', 'cache' , 'send_cached']
                        ,help='Send email, cache file into a summary, or send the summary')
    parser.add_argument('inputs', type=str, nargs='*'
                        ,help='Image files to email or cache')
    parser.add_argument('-c', '--config_file', metavar='', type=str, required=True
                        ,help='configuration file')
    parser.add_argument('-o', '--object', action='store_true', help='Enable object detection')

    args = parser.parse_args()
    
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read(config_file)
    cached_message_file = config.get('options', 'cached_message_file') 
    
    if args.object:
        
        # Load the configuration file
        config_details = sd.get_config_details( args.config_file )
        
        # Load the frozen detection graph and the corresponding class dictionary
        # Processing using only the RGB graph is currently hard coded    
        detection_graph = sd.load_frozen_detection_model( 
                config_details.get('object_detection' , 'rgb_frozen_model_path' )
                )
        id_to_label = sd.load_tf_dict( 
                config_details.get('object_detection' , 'rgb_dict_file' )
                )
        
        # (class_id , threshold)
        #plt_classes = [ (1,0.30) , (2,0.20) , (16,0.20) , (17,0.20) , (18,0.40) ]
        """ The classes of things we actually want to detect (as opposed to all
        the classes the frozen TensorFlow graph can detect) are indicated in the 
        configuration file along with the detection thresholds. This bit of code
        reverses the dictionary to go from class names (in the configuraiton file)
        to class IDs and then creates a list of tuples pairing each class ID with 
        its detection threshold."""
        plt_classes = []
        plt_class_ids = []
        for cls, thresh in zip( config_details.get('object_detection','det_classes').split(',') , 
            list(map(float,config_details.get('object_detection','det_thresholds').split(','))) ):
                key = next(key for key, value in list(id_to_label.items()) if value == cls.strip())
                plt_classes.append( (key , thresh) )
                plt_class_ids.append( key )
        plt_class_ids = np.asarray(plt_class_ids)
                
        # prefix and path of files for saving results
        res_file_prefix = config_details.get('object_detection','rgb_hist_prefix')
        out_dir    = config_details.get('object_detection', 'batch_detection_folder')
        
        od_email_recipient = [x.strip() for x in config.get('object_detection' , 'recipient' ).split(',')]
    
    if args.mode.upper() == 'SEND' or args.mode.upper()=='CACHE':
        input_path=[os.path.join(os.getcwd(), path) for path in args.inputs]
        
        for inp in input_path:
            # separate the file name from the path and the extension   
            fname = os.path.splitext(os.path.basename(inp))[0]
            
            # Get the directory of the file (will be useful later)
            dir_name = os.path.dirname(inp)
            
            # Get the event ID (unforuntately, have to rely on the format of the file name here
            event_id = fname.split("_")[1]
            camera_name = fname.split("_")[0]
            
            picture_file_name = os.path.join(dir_name, fname+'.webp')
            video_file_name   = os.path.join(dir_name, fname+'.mkv')
            
            if args.object:
                # Load the image and convert from BGR (CV2) to RGB (numpy and TensorFlow)
                image_np = cv2.cvtColor(cv2.imread(picture_file_name), cv2.COLOR_BGR2RGB)
                
                # Is the image really grayscale?
                if not sd.isgrayscale( image_np ):
                    # Run TensorFlow object detection
                    with detection_graph.as_default():
                        (boxes , scores, classes) = sd.tf_detect(image_np,detection_graph)
                        
                    # Store the score and class results
                    sd.store_results( os.path.join(out_dir,res_file_prefix) , classes , scores)
                
                    # Use box drawing function to apply detection threshold criteria
                    # and then to draw boxes (if appropriate)
                    (object_detected , image_np , top_object) = sd.draw_box2( image_np 
                            , boxes, scores , classes, id_to_label, plt_classes)
                    
                    if object_detected:
                        
                        # Write out the marked up image
                        out_path = os.path.join(dir_name,fname+'_sd'+'.webp')
                        cv2.imwrite( out_path
                                ,cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) )
                        
                        # re-direct to the annotated version of the picture
                        picture_file_name = out_path
            
            if args.mode.upper() == 'SEND':
                # Send a nicely formatted email
                video_link = MotionUploader(config_file).upload_video(video_file_name)
                alert_subject = camera_name + ' ' + event_id + ' at ' + \
                datetime.now().strftime("%I:%M%p on %B %d, %Y") 
                alert_message_body = '<a href=' + video_link +'>Video</a>'
                send_email( alert_subject, alert_message_body , config_file , 
                           [ picture_file_name ] )
            else:
                # Add to the set of cached detections
                video_link = MotionUploader(config_file).upload_video( 
                                video_file_name,picture_file_name)
                header_text = '<table>\n<tr>\n<th>Time</th>\n<th>Video</th></tr>\n'
                link_text = r'<tr>' \
                  +'<td style="text-align:center">' \
                  + datetime.now().strftime("%I:%M%p<br>%B %d") + '</td>' \
                  +'<td style="text-align:center">' + '<a href="' + video_link[0] \
                  + '"><img src="' \
                  + video_link[1] + '" width=400></a></td></tr>\n'
                if not os.path.exists(cached_message_file):
                    link_text = header_text+link_text
                hs = open(cached_message_file,"a+")
                hs.write(link_text)
                hs.close()
                
                if args.object and object_detected:
                    # Send out a nicely formatted email
                    
                    # Default subject
                    alert_subject = 'Object Detection for '
                    
                    # Start the body of the message with a link to the video
                    alert_message_body = '<a href=' + video_link[0] +'>Object Detected</a><br><br>'
                    
                    # List the object detected and their scores in the body of the message
                    for to in top_object:
                        alert_message_body+=( '{}\tat {:.2f}<br>'.format(to[1].upper(), to[0]))
                        
                        # If any people were detected, modify the message subject to indicate that
                        if to[1].upper() == 'PERSON':
                            alert_subject = 'PERSON Detection for '
                            
                    # Finish building the subject line
                    alert_subject+= camera_name.upper() + ' at '\
                                    + datetime.now().strftime("%I:%M%p on %B %d, %Y")
                                    
                    # Send an email with the annotated image as an attachment
                    send_email( alert_subject, alert_message_body , config_file , 
                       [ out_path ] , od_email_recipient)
                    
                    
    
    elif args.mode.upper() == 'SEND_CACHED' and os.path.exists(cached_message_file):
        footer_text = '</table>'
        with open(cached_message_file, 'r') as content_file:
          msg_body = content_file.read()
        send_email( r'Motion Summary at ' +  datetime.now().strftime("%I:%M%p on %B %d, %Y") 
        , msg_body + footer_text, config_file )
        os.remove(cached_message_file)
    
        # Remove files older then 30 days
        MotionUploader(config_file).cleanup(30,False)

if __name__ == '__main__':
  main()
