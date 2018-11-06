import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os

# Merge the audio and video in the folders
import subprocess

######################### Script below merges the audio and video for Mahnob in order to facilitate labelling   #######################
project_home_dir = os.getcwd()

# os.walk returns
for directory, subdirs, files in os.walk("./data/Sessions"):
	for subdir in subdirs:
		# Inside the session folder
		os.chdir( os.path.join(project_home_dir, directory, subdir) )
		# List the files in current directory
		audio_file = [f for f in os.listdir('.') if (os.path.isfile(f) and '.wav' in f)]  #Find the audio file
		video_file = [f for f in os.listdir('.') if (os.path.isfile(f) and '.avi' in f)]	 # Find the video file
		# Some files don't have both audio and video
		if len(audio_file) > 0 and len(video_file) > 0:
			mixed_file = str(audio_file[0]+ '_' + video_file[0].replace('.avi', '.mkv')) # Note merged video + audio format must be .mkv (it doesn't merge with avi)
			cmd = 'ffmpeg -y -i ' + audio_file[0] + ' -r 30 -i ' + video_file[0] + ' -filter:a aresample=async=1 -c:a flac -c:v copy ' + mixed_file
			subprocess.call(cmd, shell=True)
	# Return back to home directory
	os.chdir( project_home_dir )
	break


# Switch back to original directory
os.chdir( project_home_dir )

# Print current directory
print (os.getcwd())
#####################################################################################################################################



