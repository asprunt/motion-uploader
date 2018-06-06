#!/usr/bin/python2

import os
import re
import sys
import glob
from datetime import datetime, time
from uploader import MotionUploader
import ConfigParser

def send_email(subject, body, config_file_path, files=None):
    import smtplib

    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.utils import COMMASPACE, formatdate

    assert isinstance(files,list) | (files is None)

    config = ConfigParser.ConfigParser()
    config.read(config_file_path)
    gmail_user = config.get('gmail', 'user') 
    gmail_pwd = config.get('gmail', 'password')
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
  config_file = sys.argv[2]

  config = ConfigParser.ConfigParser()
  config.read(config_file)
  cached_message_file = config.get('options', 'cached_message_file') 

  # --send --cache --send_cached  
  if sys.argv[1].lower() == r'--send'.lower():
    input_path=sys.argv[3]
    # separate the file name from the path and the extension   
    fname = os.path.splitext(os.path.basename(input_path))[0]
    
    # Get the directory of the file (will be useful later)
    dir_name = os.path.dirname(input_path)
    
    # Get the event ID (unforuntately, have to rely on the format of the file name here
    event_id = fname.split("_")[1]
    camera_name = fname.split("_")[0]
    
    picture_file_name = os.path.join(dir_name, fname+'.webp')
    video_file_name   = os.path.join(dir_name, fname+'.mkv')
    alert_subject = camera_name + ' ' + event_id + ' at ' + \
        datetime.now().strftime("%I:%M%p on %B %d, %Y") 
    video_link = MotionUploader(config_file).upload_video(video_file_name) 
    alert_message_body = '<a href=' + video_link +'>Video</a>'
    send_email( alert_subject, alert_message_body , config_file , 
               [ picture_file_name ] )
  elif sys.argv[1].lower() == r'--cache'.lower():
    header_text = '<table>\n<tr>\n<th>Time</th>\n<th>Video</th></tr>\n'
    input_path=sys.argv[3]
    # separate the file name from the path and the extension   
    fname = os.path.splitext(os.path.basename(input_path))[0]
    
    # Get the directory of the file (will be useful later)
    dir_name = os.path.dirname(input_path)
    
    video_file_name   = os.path.join(dir_name, fname+'.mkv')
    picture_file_name = os.path.join(dir_name, fname+'.webp')
    video_link = MotionUploader(config_file).upload_video(video_file_name,picture_file_name) 
    link_text = r'<tr>' \
      +'<td style="text-align:center">' + datetime.now().strftime("%I:%M%p<br>%B %d") + '</td>' \
      +'<td style="text-align:center">' + '<a href="' + video_link[0] + '"><img src="' \
      + video_link[1] + '" width=400></a></td></tr>\n'
    if not os.path.exists(cached_message_file):
        link_text = header_text+link_text
    hs = open(cached_message_file,"a+")
    hs.write(link_text)
    hs.close()
    
  elif sys.argv[1].lower() == r'--send_cached'.lower() and os.path.exists(cached_message_file):
    footer_text = '</table>'
    with open(cached_message_file, 'r') as content_file:
      msg_body = content_file.read()
    send_email( r'Motion Summary at ' +  datetime.now().strftime("%I:%M%p on %B %d, %Y") 
    , msg_body + footer_text, config_file )
    os.remove(cached_message_file)
  else:
    print('Unexpected switch')
  


if __name__ == '__main__':
  main()
