import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os
import shutil

import keras
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing import image

import cv2  # Computer Vision
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import imageio
imageio.plugins.ffmpeg.download()

import h5py

# Keep randomness the same
np.random.seed(0)



fps = 25 # original video recorded at 25fps
num_frames_to_use = 25  # Only use 20 frames from the laughter segment
img_size = 70  # Downsample the frames to 60x60 each, originally its (576, 720)
num_classes = 5


input_data = []

# Save home directory
project_home_dir = os.getcwd()

# Empty Annotations File
df_annotations = pd.DataFrame(index=range(0,0), columns=['Type', 'Start Time (sec)', 'End Time (sec)'])

main_labelling_annotations = []


for directory, subdirs, files in os.walk("./data/Sessions"):
	# convert the subdirs name (strings) to int to sort folder numbers them so our models have same order input
	subdirs = [int(i) for i in subdirs]
	subdirs.sort()
	# convert back to string
	subdirs = [str(i) for i in subdirs]
	for subdir in subdirs:
		# Inside the session folder
		os.chdir( os.path.join(project_home_dir, directory, subdir) )
		
		# List the files in current directory
		video_file = [f for f in os.listdir('.') if (os.path.isfile(f) and '.avi' in f)]# Find the video file (might be empty)

		df_annotation = pd.read_csv('laughterAnnotation.csv', encoding="ISO-8859-1")
		
		# Only include "Laughter" or "PosedLaughter" and only columns [ Type  Start Time (sec)  End Time (sec)]
		df_annotation = df_annotation.loc[(df_annotation['Type'] == 'Laughter') | (df_annotation['Type']=='PosedLaughter')].iloc[:, 1:4]
		# Concatenate it to the main annotations file
		df_annotations = pd.concat([df_annotations,df_annotation])

		# Portion to test Laura's code
		df_temp = pd.read_csv('laughterAnnotation.csv', encoding="ISO-8859-1")
		for index, row in enumerate( np.array(df_temp) ):
			type=row[1]
			if type == 'Laughter' or type == 'PosedLaughter':
				main_labelling_annotations.append( video_file[0].strip('.avi') + '-l' + str(index).zfill(3)  )


		# Go through the annotations for this audio file only
		for index,row in enumerate( np.array(df_annotation) ):
			# row is:  	[Type   start_time	end_time]
			start_time = row[1]
			end_time = row[2]
			#Read each video and edit it
			editted_vid = VideoFileClip(video_file[0]).subclip(start_time, end_time)
			# Write it back to that folder
			editted_name = str(start_time) + "_" + str(end_time) + ".mp4"
			editted_vid.write_videofile(editted_name)
			# Read the editted video with openCV
			vidcap = cv2.VideoCapture(editted_name)
			# fps = int(vidcap.get(cv2.CAP_PROP_FPS))
			num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
			# Read all the frames 
			# 2 pre-processing after reading: resize to 70 x 70 and also convert it to GrayScale
			# image_frames = [ cv2.cvtColor( cv2.resize(vidcap.read()[1], (img_size, img_size)) , cv2.COLOR_BGR2GRAY) for i in range(0, int(num_frames)) ]   # Grayscale
			# image_frames = [ cv2.cvtColor( cv2.resize(vidcap.read()[1], (img_size, img_size)) , cv2.COLOR_BGR2GRAY).reshape(img_size,img_size,1) for i in range(0, int(num_frames)) ]    # Grayscale
			image_frames = [ cv2.resize(vidcap.read()[1], (img_size, img_size))  for i in range(0, int(num_frames)) ]   #RGB 

			# If the audio size is bigger than the desired number of samples, downsample it
			if len(image_frames) > num_frames_to_use:
				# Downsample the # frames to 25 frames
				downsampled_frames = [ image_frames[ int(np.floor(i)) ] for i in np.linspace(0,len(image_frames)-1, num_frames_to_use)]

			#Just pad the end with zeros (black frames)
			else:
				w, h, c = image_frames[0].shape  #width, height, channel
				padded_frames = [np.zeros((w,h,c)) for i in range(0, num_frames_to_use-len(image_frames))]
				downsampled_frames = image_frames + padded_frames
			
			downsampled_frames = np.array(downsampled_frames)
			input_data.append( downsampled_frames )
			# Remove the editted video
			os.remove(editted_name)

	# Return back to home directory
	os.chdir( project_home_dir )
	break


# A single video/data sample is 4D: (frames, width, height, channels)  Our channel is 1 for grayscale
x_train = np.array(input_data)

# Scale and normalize the values, divide by 255
x_train = x_train/255



# image * 255 if the pixel values have been scaled
count = 0
while success:
	image = cv2.resize(image, (img_size, img_size)) 
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = image.reshape(img_size,img_size,1)   #no difference
	cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
	success,image = vidcap.read()
	count += 1



# CROP the faces to improve the model
cropped_x_train = []

for video in x_train:
	frames = []
	for ix, frame in enumerate(video):
		os.mkdir('./'+ str(ix))
		cv2.imwrite('./' + str(ix) + '/' + str(ix) +'.jpg', frame*255) 
		# Crop it: yes is to answer the command prompt to overwrite the existing image
		cmd = 'yes | autocrop -i ' + './' + str(ix) + ' -o ./' + str(ix) + ' -w 70 -H 70'
		subprocess.call(cmd, shell=True)
		# 1 is for color, 0 is for grayscale reading
		frames.append( cv2.imread('./' + str(ix) + '/' + str(ix) +'.jpg',1)  )
		shutil.rmtree('./' + str(ix))
	cropped_x_train.append( np.array(frames) )


cropped_x_train = np.array(cropped_x_train)

# Scale and normalize the values, divide by 255
cropped_x_train = cropped_x_train/255



# Write the data into a HDF5 File
h5f = h5py.File('Video_Features_RGB_DL.h5', 'w')
h5f = h5py.File('Cropped_Video_Features_RGB_DL.h5', 'w')
h5f = h5py.File('Video_Features_Grayscale_DL.h5', 'w')
h5f.create_dataset('dataset_features', data=cropped_x_train) 
h5f.create_dataset('dataset_features', data=x_train)  # Group Name is 'dataset_features'
h5f.close()


# Loading the File
# h5f = h5py.File('Video_Features_RGB_DL.h5','r')
# h5f = h5py.File('Cropped_Video_Features_RGB_DL.h5','r')
# h5f = h5py.File('Video_Features_Grayscale_DL.h5','r')
# group_name = list(h5f.keys())[0]
# x_train = h5f[group_name][:]
# h5f.close()









