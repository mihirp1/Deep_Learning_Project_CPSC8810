'''
This file reads the image file and implements the central class for datasets

'''

from sklearn.utils import shuffle

import numpy as NMP

import cv2 as ComputerVisionToolBox

import glob, os


class FullSetOfData(object):

  def __init__(self, data_pictures, PictureMappings, NamesOfImages, ConstructorOfClassification):

    self._CountOfInstance = data_pictures.shape[0]

    self._data_pictures = data_pictures

    self._PictureMappings = PictureMappings

    self._NamesOfImages = NamesOfImages

    self._ConstructorOfClassification = ConstructorOfClassification

    self._CountOfEpochs = 0

    self._index_in_CountOfEpoch = 0

  @property
  def data_pictures(self):
    return self._data_pictures

  @property
  def NamesOfImages(self):
    return self._NamesOfImages

  @property
  def ConstructorOfClassification(self):
    return self._ConstructorOfClassification

  @property
  def CountOfInstance(self):
    return self._CountOfInstance

  @property
  def CountOfEpochs(self):
    return self._CountOfEpochs


  @property
  def PictureMappings(self):
    return self._PictureMappings


  def FollowingSample(self, SAMPLE_SET_DIMENSION):

    """Randomly selects batches from the given dataset"""
    BEGIN = self._index_in_CountOfEpoch

    self._index_in_CountOfEpoch += SAMPLE_SET_DIMENSION

    if self._index_in_CountOfEpoch > self._CountOfInstance:
      '''
      Value incremented by 1 after each CountOfEpoch
      '''
      self._CountOfEpochs += 1

      BEGIN = 0

      self._index_in_CountOfEpoch = SAMPLE_SET_DIMENSION


    FINISH = self._index_in_CountOfEpoch

    return self._data_pictures[BEGIN:FINISH], self._PictureMappings[BEGIN:FINISH], self._NamesOfImages[BEGIN:FINISH], self._ConstructorOfClassification[BEGIN:FINISH]


def reading_training_data(train_path, DataFileSize, classes, testing_dimension):

  class FullSetsOfData(object):

    pass

  data_sets = FullSetsOfData()

  data_pictures, PictureMappings, NamesOfImages, ConstructorOfClassification = load_training_data(train_path, DataFileSize, classes)

  data_pictures, PictureMappings, NamesOfImages, ConstructorOfClassification = shuffle(data_pictures, PictureMappings, NamesOfImages, ConstructorOfClassification)  

  if isinstance(testing_dimension, float):

    testing_dimension = int(testing_dimension * data_pictures.shape[0])

  validation_data_pictures = data_pictures[:testing_dimension]

  validation_PictureMappings = PictureMappings[:testing_dimension]

  validation_NamesOfImages = NamesOfImages[:testing_dimension]

  validation_ConstructorOfClassification = ConstructorOfClassification[:testing_dimension]

  train_data_pictures = data_pictures[testing_dimension:]

  train_PictureMappings = PictureMappings[testing_dimension:]

  train_NamesOfImages = NamesOfImages[testing_dimension:]

  train_ConstructorOfClassification = ConstructorOfClassification[testing_dimension:]

  data_sets.train = FullSetOfData(train_data_pictures, train_PictureMappings, train_NamesOfImages, train_ConstructorOfClassification)

  data_sets.valid = FullSetOfData(validation_data_pictures, validation_PictureMappings, validation_NamesOfImages, validation_ConstructorOfClassification)

  return data_sets



def load_training_data(train_path, DataFileSize, classes):


    data_pictures = []

    PictureMappings = []

    NamesOfImages = []

    ConstructorOfClassification = []

    count = 0

    print('Reading and assesing training data_pictures : ')

    for fields in classes:

        index = classes.index(fields)

        #print('READING & MAPPING {} FILES (Index: {})'.format(fields, index))

        path = os.path.join(train_path, fields, '*g')

        IMAGES = glob.glob(path)

        for SINGLE_IMAGE in IMAGES:

            data_file = ComputerVisionToolBox.imread(SINGLE_IMAGE)

            data_file = ComputerVisionToolBox.resize(data_file, (DataFileSize, DataFileSize),1,1, ComputerVisionToolBox.INTER_LINEAR)

            data_file = data_file.astype(NMP.float32)

            data_file = NMP.multiply(data_file, 1.0 / 255.0)

            data_pictures.append(data_file)

            label = NMP.zeros(len(classes))

            label[index] = 1.0

            count += 1

            PictureMappings.append(label)

            flbase = os.path.basename(SINGLE_IMAGE)

            NamesOfImages.append(flbase)

            ConstructorOfClassification.append(fields)

    data_pictures = NMP.array(data_pictures)

    PictureMappings = NMP.array(PictureMappings)

    NamesOfImages = NMP.array(NamesOfImages)

    ConstructorOfClassification = NMP.array(ConstructorOfClassification)

    return data_pictures, PictureMappings, NamesOfImages, ConstructorOfClassification

  
