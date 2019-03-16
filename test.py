import tensorflow as tflow

import numpy as NMP

import os,glob,sys,argparse

import cv2 as ComputerVisionToolBox

'''
Reading database image files 

'''

dir_path = os.path.dirname(os.path.realpath(__file__))

image_path=sys.argv[1] 

filename = dir_path +'/' +image_path

DataFileSize=100

number_of_channels=3

data_pictures = []


image = ComputerVisionToolBox.imread(filename)


'''
 Resizing and Preprocessing of data_pictures
'''


image = ComputerVisionToolBox.resize(image, (DataFileSize, DataFileSize),0,0, ComputerVisionToolBox.INTER_LINEAR)

data_pictures.append(image)

data_pictures = NMP.array(data_pictures, dtype=NMP.uint8)

data_pictures = data_pictures.astype('float32')

data_pictures = NMP.multiply(data_pictures, 1.0/255.0) 

'''
Reshaping input

'''

input_set = data_pictures.reshape(1, DataFileSize,DataFileSize,number_of_channels)


tfsession1 = tflow.Session()

'''
Generating network graph

'''

TEMP_BUFFER = tflow.train.import_meta_graph('trained-model.meta')

'''
Loading Weights

'''

TEMP_BUFFER.restore(tfsession1, tflow.train.latest_checkpoint('./'))

'''
Using InBuilt Graph

'''

graph = tflow.get_default_graph()


output_estimated = graph.get_tensor_by_name("output_estimated:0")


main_input = graph.get_tensor_by_name("main_input:0") 

correct_mapping = graph.get_tensor_by_name("correct_mapping:0") 

y_test_data_pictures = NMP.zeros((1, len(os.listdir('training_data')))) 

'''
Creating Dictionary For prediction

'''

input_hash_map_testing = {main_input: input_set, correct_mapping: y_test_data_pictures}

result=tfsession1.run(output_estimated, feed_dict=input_hash_map_testing)


prediction_dictionary  = {}

prediction_dictionary = {0:"slapping",1:"punching",2:"nonbullying",3:"laughing",4:"isolation",5:"quarrel",6:"stabbing",7:"strangle",8:"gossiping",9:"pullinghair"}

MaxProbability = NMP.argmax(result)

print( prediction_dictionary[MaxProbability] )
	
