#!/usr/bin/python2

import os
import re
import sys
from shutil import copyfile
from uploader import MotionUploader


def main():
	config_file     = sys.argv[1]
	for arg in sys.argv[2:]:
		# separate the file name from the path and the extension   
		fname = os.path.splitext(os.path.basename(arg))
		snapshot_fname = fname[0].split("_")[0]+r'_snapshot'
		snapshot_fname = os.path.join(os.path.dirname(arg), snapshot_fname + fname[1])
		copyfile( arg , snapshot_fname )
		MotionUploader(config_file).upload_snapshot(snapshot_fname)

if __name__ == '__main__':
  main()