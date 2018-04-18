from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3

import numpy as np
import argparse
import gen_mutation as mutate



parser = argparse.ArgumentParser(
    description='Main function for parameter estimation')
parser.add_argument('-d' '--target_dataset', help="target dataset under test",
                    choices=['MNIST','ImageNet'], default='MNIST', type=str)
parser.add_argument('-t', '--target_model', help="target model under test",
                    choices=[1, 2, 3], default=0, type=int)
parser.add_argument('-a', '--attack_type', help="attack type",
                    choices=[0, 1, 2, 3], default=0, type=int)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('mu_number', help="number of mutation", type=int)
args = parser.parse_args()


global evaluation_root
evaluation_root = '/Users/jingyi/Documents/Evaluations/deepxplore'
dataset_root = evaluation_root + '/' + args.target_dataset
dataset_model_root = dataset_root + "/model_" + args.target_model
right_prediction_data = dataset_model_root + "/right_prediction_data"
wrong_prediction_data = dataset_model_root + "/wrong_prediction_data"
attack_data = dataset_model_root + "/attack_" + args.attack_type

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = np.concatenate((x_test, x_train), axis=0)
y_test = np.concatenate((y_test, y_train), axis=0)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

if args.target_model == 1:
    model = model1
elif args.target_model == 2:
    model = model2
elif args.target_model == 3:
    model = model3

[normal_label_change,adv_label_change] = mutate.label_change_mutation(args.step,args.mu_number, args.seeds)

print("Args: ", args)
print("Normal data label change:")
print(normal_label_change)
print('Percentage of label change for normal data: ', sum(normal_label_change)/args.seeds/args.mu_number)
print("Adv data label change:")
print(adv_label_change)
print(adv_label_change)
print('Percentage of label change for adversary data: ', sum(adv_label_change)/args.seeds/args.mu_number)




