from cleverhans_tutorials.tutorial_models import make_basic_cnn
from gen_mutation import MutationTest
import numpy as np
import tensorflow as tf
import pickle
from cleverhans.utils_tf import model_eval, model_argmax
import utils
import os
from cleverhans.utils_mnist import data_mnist
from os.path import expanduser
from adversary_detector import detector

home = expanduser("~")
dataset = 'mnist'
attack_type = 'jsma'
folder_name = dataset+'_'+attack_type

image_rows = 28
image_cols = 28

# Restore the DNN model
sess = tf.Session()
x = tf.placeholder(tf.float32, shape=(None, image_rows, image_cols, 1))
# Define TF model graph
model = make_basic_cnn()
preds = model(x)

saver = tf.train.Saver()
saved_model_path = os.path.join(home + '/cleverhans/cleverhans_tutorials/'+dataset+'_'+attack_type+'/model', attack_type+'.model')
print('--- Restoring trained model from: ', saved_model_path)
saver.restore(sess, saved_model_path)


# Define a detector
ad = detector(0.0019, 1, image_rows, image_cols, 1, 2000, 0.05, 0.05, 0.0019*0.8)
print('--- Detector config: ', ad.print_config())

# X_train, Y_train, X_test, Y_test = data_mnist(train_start=0,
#                                                   train_end=1000,
#                                                   test_start=0,
#                                                   test_end=1000)

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

adv_image_dir = home+'/cleverhans/cleverhans_tutorials/'+folder_name+'/adv_'+attack_type
print('--- Extracting adversary images from: ', adv_image_dir)
[adv_image_list, adv_image_files, real_labels, predicted_labels] = utils.get_data_mutation_test(adv_image_dir)
adv_count = 0
total_mutation_counts = []
label_change_mutation_counts = []

print('--- Evaluating adversarial inputs ---')

not_decided_images = 0
for i in range(len(adv_image_list)):
    # print('- Running image ', i)
    ori_img = np.expand_dims(adv_image_list[i], axis=2)
    ori_img = ori_img.astype('float32')
    ori_img /= 255
    orig_label = predicted_labels[i]
    [result, decided, total_mutation_count, label_change_mutation_count] = ad.detect(ori_img,orig_label, sess, x, preds)
    if result:
        adv_count += 1
    if not decided:
        not_decided_images += 1
    total_mutation_counts.append(total_mutation_count)
    label_change_mutation_counts.append(label_change_mutation_count)

print('- Total adversary images evaluated: ', len(adv_image_list))
print('- Identified adversaries: ', adv_count)
print('- Not decided images: ', not_decided_images)
print('- Average mutation needed: ', sum(total_mutation_counts)/len(total_mutation_counts))
print('- Average label change mutations: ', float(sum(label_change_mutation_counts))/len(label_change_mutation_counts))
print(total_mutation_counts)
print(label_change_mutation_counts)