from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from utils import *
import os
import random
import numpy as np

def mutation_matrix(step_size):
    method = random.randint(1, 3)
    trans_matrix = generate_value(img_rows, img_cols)
    rect_shape = (random.randint(1, 3), random.randint(1, 3))
    start_point = (
        random.randint(0, img_rows - rect_shape[0]),
        random.randint(0, img_cols - rect_shape[1]))

    if method == 1:
        transformation = c_light(trans_matrix)
    elif method == 2:
        transformation = c_occl(trans_matrix, start_point, rect_shape)
    elif method == 3:
        transformation = c_black(trans_matrix, start_point, rect_shape)

    return transformation * step_size


def split_image_train():
    '''split the training image to normal image (the predication is right)
        and adversary image (the predication is wrong) and save them to path'''


def label_change_mutation(step_size, mutation_number, seed_number):
    ''' This function returns the label change of random mutations'''

    mutations = []
    for i in range(mutation_number):
        mutation = mutation_matrix(step_size)
        mutations.append(mutation)

    diverged_true = {}
    diverged_false = {}
    true = 0
    false = 0
    i = 0
    while true < seed_number or false < seed_number:


        # check if the image has been selected, if yes, reuse them
        # force generate a new batch if force_generate is true

        # if(os.path.exists()):
        i = i + 1
        number = random.randint(0, 69999)
        # ori_img = np.expand_dims(random.choice(x_test), axis=0)
        ori_img = np.expand_dims(x_test[number], axis=0)

        # first check if input already induces differences
        ori_label = np.argmax(model.predict(ori_img)[0])
        if ori_label == y_test[number] and true < args.seeds:
            true = true + 1
            ori_img_de = ori_img.copy()
            orig_img_deprocessed = deprocess_image(ori_img_de)

            # save the result to disk
            store_path = './mutation_test/' + str(step_size) + "_" + str(mutation_number) + "/true/" + str(i)
            isExists = os.path.exists(store_path)
            if not isExists:
                os.makedirs(store_path)
            imsave(store_path + "/" + str(ori_label) + '_orig.png', orig_img_deprocessed)

            # if ori_label == y_test[i]:
            #     store_path = "./test_data/true/"
            #     isExists = os.path.exists(store_path)
            #     if not isExists:
            #         os.makedirs(store_path)
            #     imsave(store_path + "/" + str(i) + "label" + str(ori_label) + '.png', orig_img_deprocessed)
            # else:
            #     store_path = "./test_data/false/"
            #     isExists = os.path.exists(store_path)
            #     if not isExists:
            #         os.makedirs(store_path)

            #     imsave(store_path + "/" + str(i) + "label" + str(y_test[i]) + "_plabel" + str(ori_label) + '.png', orig_img_deprocessed)

            count = 0
            for j in range(mutation_number):
                img = ori_img.copy()
                mu_img = img + mutations[j]
                mu_label = np.argmax(model.predict(mu_img)[0])

                mu_img_deprocessed = deprocess_image(mu_img)
                imsave(store_path + "/" + str(j) + "_" + str(mu_label) + '.png', mu_img_deprocessed)

                if mu_label != ori_label:
                    count += 1

            diverged_true[str(i)] = count

            path = './mutation_test/' + str(step_size) + "_" + str(mutation_number) + "/true/" + str(i) + "/" + str(count)
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
        elif ori_label != y_test[number] and false < seed_number:
            false = false + 1
            # same with true
            ori_img_de = ori_img.copy()
            orig_img_deprocessed = deprocess_image(ori_img_de)

            # save the result to disk
            store_path = './mutation_test/' + str(step_size) + "_" + str(mutation_number) + "/false/" + str(i)
            isExists = os.path.exists(store_path)
            if not isExists:
                os.makedirs(store_path)
            imsave(store_path + "/" + str(ori_label) + '_orig.png', orig_img_deprocessed)

            # if ori_label == y_test[i]:
            #     store_path = "./test_data/true/"
            #     isExists = os.path.exists(store_path)
            #     if not isExists:
            #         os.makedirs(store_path)
            #     imsave(store_path + "/" + str(i) + "label" + str(ori_label) + '.png', orig_img_deprocessed)
            # else:
            #     store_path = "./test_data/false/"
            #     isExists = os.path.exists(store_path)
            #     if not isExists:
            #         os.makedirs(store_path)
            #     imsave(store_path + "/" + str(i) + "label" + str(y_test[i]) + "_plabel" + str(ori_label) + '.png', orig_img_deprocessed)

            count = 0
            for j in range(mutation_number):
                img = ori_img.copy()
                mu_img = img + mutations[j]
                mu_label = np.argmax(model.predict(mu_img)[0])

                mu_img_deprocessed = deprocess_image(mu_img)
                imsave(store_path + "/" + str(j) + "_" + str(mu_label) + '.png', mu_img_deprocessed)

                if mu_label != ori_label:
                    count += 1

            diverged_false[str(i)] = count

            path = './mutation_test/' + str(step_size) + "_" + str(mutation_number) + "/false/" + str(i) + "/" + str(
                count)
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)

    true_nums = []
    false_nums = []

    # print("true:", diverged_true)
    for dt in diverged_true:
        true_nums.append(diverged_true[dt])

    # print(true_nums)

    # print("false:", diverged_false)
    for df in diverged_false:
        false_nums.append(diverged_false[df])
    # print(false_nums)

    return true_nums,false_nums


parser = argparse.ArgumentParser(
    description='Main function for mutation algorithm for input generation in Driving dataset')
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('mu_number', help="number of mutation", type=int)
args = parser.parse_args()

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

if args.target_model == 0:
    model = model1
elif args.target_model == 1:
    model = model2
elif args.target_model == 2:
    model = model3

[normal_label_change,adv_label_change] = label_change_mutation(args.step,args.mu_number, args.seeds)

print("Args: ", args)
print("Normal data label change:")
print(normal_label_change)
print("Adv data label change:")
print(adv_label_change)

