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
import tensorflow as tf
from cleverhans.utils_tf import model_eval, model_argmax
from cleverhans_tutorials.tutorial_models import make_basic_cnn
import pickle


class MutationTest:

    '''
        Mutation testing for the training dataset
        :param img_rows:
        :param img_cols:
        :param step_size:
        :param mutation_number:
    '''

    img_rows = 28
    img_cols = 28
    step_size = 1
    seed_number = 500
    mutation_number = 1000

    def __init__(self, img_rows, img_cols, step_size, seed_number, mutation_number):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.step_size = step_size
        self.seed_number = seed_number
        self.mutation_number = mutation_number


    # def split_image_train(model_number):
    #     '''split the training image to normal image (the predication is right)
    #         and adversary image (the predication is wrong) and save them to path'''
    #
    #     pos_number = 0
    #     neg_number = 0
    #     for number in range(0,70000):
    #         ori_img = np.expand_dims(x_test[number], axis=0)
    #         ori_label = np.argmax(model.predict(ori_img)[0])
    #         ori_img_de = ori_img.copy()
    #         orig_img_deprocessed = deprocess_image(ori_img_de)
    #
    #         if ori_label == y_test[number]:
    #             pos_number = pos_number + 1
    #             # save the result to disk
    #             store_path = '/Users/jingyi/Documents/Evaluations/deepxplore/MNIST/model_' + str(model_number) + '/right_prediction_data'
    #             isExists = os.path.exists(store_path)
    #             if not isExists:
    #                 os.makedirs(store_path)
    #             imsave(store_path + "/" + str(number) + '_orig.png', orig_img_deprocessed)
    #         elif ori_label != y_test[number]:
    #             neg_number = neg_number + 1
    #             # save the result to disk
    #             store_path = '/Users/jingyi/Documents/Evaluations/deepxplore/MNIST/model_' + str(model_number) + '/wrong_prediction_data'
    #             isExists = os.path.exists(store_path)
    #             if not isExists:
    #                 os.makedirs(store_path)
    #             imsave(store_path + "/" + str(number) + '_orig.png', orig_img_deprocessed)
    #
    #     print('Number of positive data: ', pos_number)
    #     print('Number of negative data: ', neg_number)

    def mutation_matrix(self):

        method = random.randint(1, 3)
        trans_matrix = generate_value(self.img_rows, self.img_cols)
        rect_shape = (random.randint(1, 3), random.randint(1, 3))
        start_point = (
            random.randint(0, self.img_rows - rect_shape[0]),
            random.randint(0, self.img_cols - rect_shape[1]))

        if method == 1:
            transformation = c_light(trans_matrix)
        elif method == 2:
            transformation = c_occl(trans_matrix, start_point, rect_shape)
        elif method == 3:
            transformation = c_black(trans_matrix, start_point, rect_shape)

        return transformation

    def label_change_mutation_test(self, sess, test_data, orig_labels):

        '''
        :param model: the model under defense
        :param test_data: test data for label changes
        :param orig_labels: original labels predicted by the model
        :param step_size: step size of the mutation
        :param mutation_number: number of mutations
        :return: label changes in the given number of mutations
        '''

        # Generate random matution matrix for mutations
        mutations = []
        for i in range(self.mutation_number):
            mutation = self.mutation_matrix()
            mutations.append(mutation)

        label_change_numbers = []

        # Iterate over all the test data
        for i in range(len(orig_labels)):
            ori_img = np.expand_dims(test_data[i], axis=2)
            orig_label = orig_labels[i]

            label_changes = 0
            for j in range(self.mutation_number):
                img = ori_img.copy()
                add_mutation = mutations[j][0]
                mu_img = img + add_mutation

                # Predict the label for the mutation
                mu_img = np.expand_dims(mu_img, 0)


                # Define input placeholder
                input_x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))


                # Load the symbolic output
                with open('/Users/jingyi/cleverhans-master/cleverhans_tutorials/preds.pkl', "rb") as f:
                    preds = pickle.load(f)

                mu_label = model_argmax(sess, input_x, preds, mu_img)
                print('Predicted label: ', mu_label)

                if mu_label!=orig_label:
                    label_changes += 1

            label_change_numbers.append(label_changes)

        print('Number of label changes: ', label_change_numbers)
        return label_change_numbers


    def label_change_mutation_train(self):
        ''' This function returns the label change of random mutations in the training data'''

        mutations = []
        for i in range(self.mutation_number):
            mutation = self.mutation_matrix()
            mutations.append(mutation)

        diverged_true = {}
        diverged_false = {}
        true = 0
        false = 0
        i = 0
        while true < self.seed_number or false < self.seed_number:
            # check if the image has been selected, if yes, reuse them
            # force generate a new batch if force_generate is true

            # if(os.path.exists()):
            i = i + 1
            # number = random.randint(0, 69999)
            number = i
            # ori_img = np.expand_dims(random.choice(x_test), axis=0)
            ori_img = np.expand_dims(x_test[number], axis=0)

            # first check if input already induces differences
            ori_label = np.argmax(model.predict(ori_img)[0])
            if ori_label == y_test[number] and true < self.seed_number:
                true = true + 1
                ori_img_de = ori_img.copy()
                orig_img_deprocessed = deprocess_image(ori_img_de)

                # save the result to disk
                store_path = evaluation_root + '/mutation_test/' + str(self.step_size) + "_" + str(self.mutation_number) + "/true/" + str(i)
                isExists = os.path.exists(store_path)
                if not isExists:
                    os.makedirs(store_path)
                imsave(store_path + "/" + str(ori_label) + '_orig.png', orig_img_deprocessed)

                count = 0
                for j in range(self.mutation_number):
                    img = ori_img.copy()
                    mu_img = img + mutations[j]
                    mu_label = np.argmax(model.predict(mu_img)[0])

                    mu_img_deprocessed = deprocess_image(mu_img)
                    imsave(store_path + "/" + str(j) + "_" + str(mu_label) + '.png', mu_img_deprocessed)

                    if mu_label != ori_label:
                        count += 1

                diverged_true[str(i)] = count

                path = evaluation_root + '/mutation_test/' + str(self.step_size) + "_" + str(self.mutation_number) + "/true/" + str(i) + "/" + str(count)
                isExists = os.path.exists(path)
                if not isExists:
                    os.makedirs(path)
            elif ori_label != y_test[number] and false < self.seed_number:
                false = false + 1
                # same with true
                ori_img_de = ori_img.copy()
                orig_img_deprocessed = deprocess_image(ori_img_de)

                # save the result to disk
                store_path = evaluation_root + '/mutation_test/' + str(self.step_size) + "_" + str(self.mutation_number) + "/false/" + str(i)
                isExists = os.path.exists(store_path)
                if not isExists:
                    os.makedirs(store_path)
                imsave(store_path + "/" + str(ori_label) + '_orig.png', orig_img_deprocessed)

                count = 0
                for j in range(self.mutation_number):
                    img = ori_img.copy()
                    mu_img = img + mutations[j]
                    mu_label = np.argmax(model.predict(mu_img)[0])

                    mu_img_deprocessed = deprocess_image(mu_img)
                    imsave(store_path + "/" + str(j) + "_" + str(mu_label) + '.png', mu_img_deprocessed)

                    if mu_label != ori_label:
                        count += 1

                diverged_false[str(i)] = count

                path = evaluation_root + '/mutation_test/' + str(self.step_size) + "_" + str(self.mutation_number) + "/false/" + str(i) + "/" + str(
                    count)
                isExists = os.path.exists(path)
                if not isExists:
                    os.makedirs(path)

        true_nums = []
        false_nums = []

        print('True label changes: ', diverged_true)
        for dt in diverged_true:
            true_nums.append(diverged_true[dt])

        print('False label changes: ', diverged_false)
        for df in diverged_false:
            false_nums.append(diverged_false[df])

        return true_nums,false_nums


# parser = argparse.ArgumentParser(
#     description='Main function for mutation algorithm for input generation in Driving dataset')
# parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
#                     choices=[1, 2, 3], default=1, type=int)
# parser.add_argument('-a', '--attack_type', help="attack type",
#                     choices=[0, 1, 2, 3], default=0, type=int)
# parser.add_argument('seed_number', help="number of seeds of input", type=int)
# parser.add_argument('step_size', help="step size of gradient descent", type=float)
# parser.add_argument('mutation_number', help="number of mutation", type=int)
# args = parser.parse_args()
#
# evaluation_root = '/Users/jingyi/Documents/Evaluations/deepxplore/MNIST/model_' + \
#                   str(args.target_model) + '/attack_' + str(args.attack_type)
#
# # input image dimensions
# img_rows, img_cols = 28, 28
# # the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_test = np.concatenate((x_test, x_train), axis=0)
# y_test = np.concatenate((y_test, y_train), axis=0)
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# input_shape = (img_rows, img_cols, 1)
#
# x_test = x_test.astype('float32')
# x_test /= 255
#
# # define input tensor as a placeholder
# input_tensor = Input(shape=input_shape)
#
# # load multiple models sharing same input tensor
# model1 = Model1(input_tensor=input_tensor)
# model2 = Model2(input_tensor=input_tensor)
# model3 = Model3(input_tensor=input_tensor)
#
# if args.target_model == 1:
#     model = model1
# elif args.target_model == 2:
#     model = model2
# elif args.target_model == 3:
#     model = model3
#
# # split_image_train(args.target_model)
# mt = MutationTest(img_rows, img_cols, args.step_size, args.seed_number, args.mutation_number)
# [normal_label_change,adv_label_change] = mt.label_change_mutation_train()
# print("Args: ", args)
# print("Normal data label change:")
# print(normal_label_change)
# print('Sum of label change for normal data: ', sum(normal_label_change))
# print("Adv data label change:")
# print(adv_label_change)
# print('Sum of label change for adversary data: ', sum(adv_label_change))
