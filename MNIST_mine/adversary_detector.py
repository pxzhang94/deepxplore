from cleverhans_tutorials.tutorial_models import make_basic_cnn
from gen_mutation import MutationTest
import numpy as np
import tensorflow as tf
import pickle
from cleverhans.utils_tf import model_eval, model_argmax
import utils
import os
from cleverhans.utils_mnist import data_mnist
class detector:

    '''
        A statistical detector for adversaries
        :param alpha : error bound of false negative
        :param beta : error bound of false positive
        :param sigma : size of indifference region
        :param kappa_nor : ratio of label change of a normal input
        :param mu : hyper parameter reflecting the difference between kappa_nor and kappa_adv
    '''

    alpha = 0.05
    beta = 0.05
    sigma = 0.05
    kappa_nor = 0.01
    mu = 2
    img_rows = 28
    img_cols = 28
    step_size = 1
    max_mutation = 5000

    def __init__(self, kappa_nor, mu, img_rows, img_cols, step_size, max_mutation=5000, alpha=0.05, beta=0.05, sigma=0.01):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.kappa_nor = kappa_nor
        self.mu = mu
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.step_size = step_size
        self.max_mutation = max_mutation
        assert self.kappa_nor > self.sigma

    def calculate_sprt_ratio(self, c, n):
        '''
        :param c: number of label changes
        :param n: total number of mutations
        :return: the sprt ratio
        '''

        p1 = self.mu * self.kappa_nor + self.sigma
        p0 = self.mu * self.kappa_nor - self.sigma

        return pow(p1,c)*pow(1-p1,n-c)/pow(p0,c)/pow(1-p0,n-c)


    def detect(self, orig_img, orig_label):
        stop = False
        label_change_mutation_count = 0
        total_mutation_count = 0

        while (not stop):
            total_mutation_count += 1

            if total_mutation_count>self.max_mutation:
                # print('====== Result: Can\'t make a decision in ' + str(total_mutation_count-1) + ' mutations')
                # print('Total number of mutations evaluated: ', total_mutation_count-1)
                # print('Total label changes of the mutations: ', label_change_mutation_count)
                # print('=======')
                return False, total_mutation_count, label_change_mutation_count

            # print('Test mutation number ', i)
            mt = MutationTest(self.img_rows, self.img_cols, self.step_size)
            mutation_matrix = mt.mutation_matrix()
            mutation_img = orig_img + mutation_matrix * self.step_size

            total_mutation_count += 1

            mu_label = model_argmax(sess, x, preds, mutation_img)

            if orig_label!=mu_label:
                label_change_mutation_count += 1

            sprt_ratio = self.calculate_sprt_ratio(label_change_mutation_count,total_mutation_count)

            if sprt_ratio >= (1-self.beta)/self.alpha:
                # print('=== Result: Adversarial input ===')
                # print('Total number of mutations evaluated: ', total_mutation_count)
                # print('Total label changes of the mutations: ', label_change_mutation_count)
                # print('======')
                return True, total_mutation_count, label_change_mutation_count

            elif sprt_ratio<= self.beta/(1-self.alpha):
                # print('=== Result: Normal input ===')
                # print('Total number of mutations evaluated: ', total_mutation_count)
                # print('Total label changes of the mutations: ', label_change_mutation_count)
                # print('======')
                return False, total_mutation_count, label_change_mutation_count


# Restore the DNN model
sess = tf.Session()
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
# Define TF model graph
model = make_basic_cnn()
preds = model(x)

saver = tf.train.Saver()
saver.restore(
    sess, os.path.join(
        '/Users/jingyi/cleverhans/cleverhans_tutorials/model', 'jsma.model'))


# Define a detector
ad = detector(0.003, 1, 28, 28, 1, 1000, 0.05, 0.05, 0.0025)

X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
                                                  train_end=1000,
                                                  test_start=0,
                                                  test_end=1000)

# nor_image_list = []
# nor_labels = []
#
# for i in range(len(X_train)):
#     nor_label = np.argmax(Y_train[i])
#     nor_image = np.expand_dims(X_train[i],0)
#     predicted_label = model_argmax(sess, x, preds, nor_image)
#     if nor_label==predicted_label:
#         nor_image_list.append(nor_image)
#         nor_labels.append(nor_label)


# print('--- Evaluating normal inputs ---')
#
# nor_count = 0
# total_mutation_counts_nor = []
# label_change_mutation_counts_nor = []
# for i in range(len(nor_image_list)):
#     if i>5:
#         break
#     ori_img = nor_image_list[i]
#     orig_label = nor_labels[i]
#     [result, total_mutation_count_nor, label_change_mutation_count_nor] = ad.detect(ori_img,orig_label)
#     if not result:
#         nor_count += 1
#     total_mutation_counts_nor.append(total_mutation_count_nor)
#     label_change_mutation_counts_nor.append(label_change_mutation_count_nor)
#
# print('- Total normal images evaluated: ', len(total_mutation_counts_nor))
# print('- Identified normals: ', nor_count)
# print('- Average mutation needed: ', sum(total_mutation_counts_nor)/len(total_mutation_counts_nor))
# print('- Average label change mutations: ', float(sum(label_change_mutation_counts_nor))/len(label_change_mutation_counts_nor))
# print(total_mutation_counts_nor)
# print(label_change_mutation_counts_nor)


[adv_image_list, real_labels, predicted_labels] = utils.get_data_mutation_test('/Users/jingyi/cleverhans/cleverhans_tutorials/adv_jsma')
adv_count = 0
total_mutation_counts = []
label_change_mutation_counts = []

print('--- Evaluating adversarial inputs ---')

for i in range(len(adv_image_list)):
    ori_img = np.expand_dims(adv_image_list[i], axis=2)
    ori_img = ori_img.astype('float32')
    ori_img /= 255
    orig_label = predicted_labels[i]
    [result, total_mutation_count, label_change_mutation_count] = ad.detect(ori_img,orig_label)
    if result:
        adv_count += 1
    total_mutation_counts.append(total_mutation_count)
    label_change_mutation_counts.append(label_change_mutation_count)

print('- Total images evaluated: ', len(adv_image_list))
print('- Identified adversaries: ', adv_count)
print('- Average mutation needed: ', sum(total_mutation_counts)/len(total_mutation_counts))
print('- Average label change mutations: ', float(sum(label_change_mutation_counts))/len(label_change_mutation_counts))
print(total_mutation_counts)
print(label_change_mutation_counts)



