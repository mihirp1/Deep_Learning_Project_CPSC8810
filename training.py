import reading_data, time, math, time, random,os

import tensorflow as tflow

from datetime import timedelta

import numpy as NMP

from numpy.random import seed

from tensorflow import set_random_seed


'''
Making use of randomization for making the datasets truly random
'''

seed(1)

set_random_seed(2)

SAMPLE_SET_DIMENSION = 4

#Prepare input data

classes = os.listdir('training_data')

number_of_classes = len(classes)

'''
Setting percentage chunk of dataset that will be used for validation
'''

testing_dimension = 0.20

dimension_of_im_file = 100

number_of_channels = 3

train_path='training_data'

'''
OPencv helps in loading and image operations
'''

data = reading_data.reading_training_data(train_path, dimension_of_im_file, classes, testing_dimension=testing_dimension)


print("The Data has succesfullly been read and analysed : ")

print("Image Count in Training :\t{}".format(len(data.train.PictureMappings)))

print("Image count in Testing:\t{}".format(len(data.valid.PictureMappings)))


session = tflow.Session()

main_input = tflow.placeholder(tflow.float32, shape=[None, dimension_of_im_file,dimension_of_im_file,number_of_channels], name='main_input')

'''
Creating PictureMappings for dataset
'''

correct_mapping = tflow.placeholder(tflow.float32, shape=[None, number_of_classes], name='correct_mapping')

correct_mapping_cls = tflow.argmax(correct_mapping, dimension=1)


'''
Convolution filetrs setup and parameters
'''


convolution_filter_dimension1 = 3 

number_of_convolution_filters1 = 32

convolution_filter_dimension2 = 3

number_of_convolution_filters2 = 32

convolution_filter_dimension3 = 3

number_of_convolution_filters3 = 64
    
full_convolutional_layer_size = 128


def generate_weighted_vectors(shape):

    return tflow.Variable(tflow.truncated_normal(shape, stddev=0.05))

def generate_input_bias_values(size):

    return tflow.Variable(tflow.constant(0.05, shape=[size]))




def convolution_layer_implementation(input,
               num_input_channels, 
               conv_filter_size,        
               filter_c):  
    
    '''
    Define Weights for training
    '''

    weighted_vectors = generate_weighted_vectors(shape=[conv_filter_size, conv_filter_size, num_input_channels, filter_c])

    '''
    Bias Implementation
    '''

    input_biases = generate_input_bias_values(filter_c)

    '''
    Convolution Layer Design and Generation
    '''
    layer = tflow.nn.conv2d(input=input,
                     filter=weighted_vectors,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += input_biases

    '''
    Max-Pooling
    '''

    layer = tflow.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    
    layer = tflow.nn.relu(layer)

    return layer

    

def flat_layer_generate(layer):
    '''
    Using Last layer

    '''
    layer_shape = layer.get_shape()
    '''
    ## Total feature count = Height * Width * Channel Count
    '''
    feature_count = layer_shape[1:4].num_elements()

    layer = tflow.reshape(layer, [-1, feature_count])

    return layer


def full_convolution_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    weighted_vectors = generate_weighted_vectors(shape=[num_inputs, num_outputs])

    input_biases = generate_input_bias_values(num_outputs) 

    layer = tflow.matmul(input, weighted_vectors) + input_biases

    if use_relu:

        layer = tflow.nn.relu(layer)

    return layer


layer_conv1 = convolution_layer_implementation(input=main_input,
               num_input_channels=number_of_channels,

               conv_filter_size=convolution_filter_dimension1,

               filter_c=number_of_convolution_filters1)
layer_conv2 = convolution_layer_implementation(input=layer_conv1,

               num_input_channels=number_of_convolution_filters1,

               conv_filter_size=convolution_filter_dimension2,

               filter_c=number_of_convolution_filters2)

layer_conv3= convolution_layer_implementation(input=layer_conv2,

               num_input_channels=number_of_convolution_filters2,

               conv_filter_size=convolution_filter_dimension3,

               filter_c=number_of_convolution_filters3)
          
layer_flat = flat_layer_generate(layer_conv3)

layer_fc1 = full_convolution_layer(input=layer_flat,

                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),

                     num_outputs=full_convolutional_layer_size,

                     use_relu=True)

layer_fc2 = full_convolution_layer(input=layer_fc1,

                     num_inputs=full_convolutional_layer_size,

                     num_outputs=number_of_classes,

                     use_relu=False) 

output_estimated = tflow.nn.softmax(layer_fc2,name='output_estimated')

output_estimated_cls = tflow.argmax(output_estimated, dimension=1)

session.run(tflow.initialize_all_variables())

entropy_value = tflow.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=correct_mapping)
expenditure_lost = tflow.reduce_mean(entropy_value)

network_optimizer = tflow.train.AdamOptimizer(learning_rate=1e-4).minimize(expenditure_lost)

correct_prediction = tflow.equal(output_estimated_cls, correct_mapping_cls)

resultant_accuracy = tflow.reduce_mean(tflow.cast(correct_prediction, tflow.float32))

session.run(tflow.initialize_all_variables()) 


def display_status(CountOfEpoch, input_hash_map_train, input_hash_map_validate, testing_loss):

    acc = session.run(resultant_accuracy, feed_dict=input_hash_map_train)

    testing_accuracy = session.run(resultant_accuracy, feed_dict=input_hash_map_validate)

    #msg = "Epoch Count  {0} --- Training Accuracy: {1:>6.1%}, Testing Accuracy: {2:>6.1%},  Testing Loss: {3:.3f}"

    print("Training The Network..This might take a while..")

    #print(msg.format(CountOfEpoch + 1, acc, testing_accuracy, testing_loss))

all_i_count = 0

TEMP_BUFFER = tflow.train.Saver()

def deep_network_train(num_iteration):

    global all_i_count
    
    for i in range(all_i_count,
                   all_i_count + num_iteration):

        input_set, output_correct_set, _, category_batch = data.train.FollowingSample(SAMPLE_SET_DIMENSION)

        input_correct_set, output_val_set, _, valid_category_batch = data.valid.FollowingSample(SAMPLE_SET_DIMENSION)

        
        input_hash_map_tr = {main_input: input_set,
                           correct_mapping: output_correct_set}
        input_hash_map_val = {main_input: input_correct_set,
                              correct_mapping: output_val_set}

        session.run(network_optimizer, feed_dict=input_hash_map_tr)

        if i % int(data.train.CountOfInstance/SAMPLE_SET_DIMENSION) == 0: 
            testing_loss = session.run(expenditure_lost, feed_dict=input_hash_map_val)

            CountOfEpoch = int(i / int(data.train.CountOfInstance/SAMPLE_SET_DIMENSION))    
            
            display_status(CountOfEpoch, input_hash_map_tr, input_hash_map_val, testing_loss)

            TEMP_BUFFER.save(session, 'trained-model') 


    all_i_count += num_iteration





deep_network_train(num_iteration=8500)
