import numpy as np
import h5py
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os

# Audio  MFCC Package
import librosa

# Merge the audio and video in the folders
import subprocess

# Keep randomness the same
np.random.seed(0)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.height', 1000)
pd.set_option('display.width', 1000)


# Downsample all the laughs to 2.5 seconds = (44100 samples/sec * 2.5)   @44KHz
num_samples = 110250  # Each audio file should have length of 110,250 samples
num_MFCC = 20  # Number of MFCC features  (this will result in each audio having 20x216 input shape)

input_data = []


# Save home directory
project_home_dir = os.getcwd()

# Empty Annotations File
df_annotations = pd.DataFrame(index=range(0,0), columns=['Type', 'Start Time (sec)', 'End Time (sec)'])


for directory, subdirs, files in os.walk("./data/Sessions"):
	# convert the subdirs name (strings) to int to sort folder numbers so our models have same order input
	subdirs = [int(i) for i in subdirs]
	subdirs.sort()
	# convert back to string
	subdirs = [str(i) for i in subdirs]
	for subdir in subdirs:
		# Inside the session folder
		os.chdir( os.path.join(project_home_dir, directory, subdir) )
		
		# List the files in current directory
		audio_file = [f for f in os.listdir('.') if (os.path.isfile(f) and '.wav' in f)]  #Find the audio file  (might be empty)
		
		df_annotation = pd.read_csv('laughterAnnotation.csv', encoding="ISO-8859-1")
		
		# Only include "Laughter" or "PosedLaughter" and only columns [ Type  Start Time (sec)  End Time (sec)]
		df_annotation = df_annotation.loc[(df_annotation['Type'] == 'Laughter') | (df_annotation['Type']=='PosedLaughter')].iloc[:, 1:4]
		# Concatenate it to the main annotations file
		df_annotations = pd.concat([df_annotations,df_annotation])
		
		# Go through the annotations for this audio file only
		for row in np.array(df_annotation):
			# row is:  	[Type   start_time	end_time]
			start_time = row[1]
			end_time = row[2]
			# Load "duration" seconds of a wav file, starting "offset" seconds in
			# y is the audio time series samples
			# sr is the sampling rate at which the audio file was samples  (default sr=22050). 
			y, sr = librosa.load(audio_file[0], offset=start_time, duration=(end_time-start_time), sr=None)
			# If the audio size is bigger than the desired number of samples, downsample it
			if len(y) > num_samples:
				downsampled_audio = [ y[ int(np.floor(i)) ] for i in np.linspace(0,len(y)-1, num_samples)]
			#Just pad the end with zeros
			else:
				padded_zeros = [0 for i in range(0, num_samples-len(y))]
				downsampled_audio = list(y) + padded_zeros
			downsampled_audio = np.array(downsampled_audio)
			# Find MFCC features
			# y is the audio time series and sr is the sampling rate of y, 
			MFCCs  = librosa.feature.mfcc(y=downsampled_audio, sr=sr, n_mfcc=num_MFCC)
			#20ms window (frame) with 10ms stride (overlap is 10ms)
			# print (librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20,hop_length=int(0.010*sr), n_fft=int(0.020*sr) ).shape)
			input_data.append( MFCCs )
	# Return back to home directory
	os.chdir( project_home_dir )
	break


x_train = np.array(input_data)




##################### Single Audio File Processing Example #######################
# y, sr = librosa.load("./data/audio/2.wav", sr=None, offset=0, duration=(2.6-0))
# print (librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).shape)
# # If the audio size is bigger than the number of samples, downsample it
# if len(y) > num_samples:
# 	downsampled_audio = [ y[ int(np.floor(i)) ] for i in np.linspace(0,len(y)-1, num_samples)]
# #Just pad the end with zeros
# else:
# 	padded_zeros = [0 for i in range(0, num_samples-len(y))]
# 	downsampled_audio = list(y) + padded_zeros
# print (librosa.feature.mfcc(y=np.array(downsampled_audio), sr=sr, n_mfcc=20).shape)
##################################################################################



# LSTM expects timesteps x features  (ie. 216 x 20 MFCC features)
x_train = np.swapaxes(x_train,1,2)


# Write the data into a HDF5 File
h5f = h5py.File('Audio_Features_DL.h5', 'w')
h5f.create_dataset('dataset_features', data=x_train)
h5f.close()















