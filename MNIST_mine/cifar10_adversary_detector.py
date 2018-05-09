from cleverhans_tutorials.tutorial_models import make_basic_cnn
from gen_mutation import MutationTest
import numpy as np
import tensorflow as tf
import pickle
from cleverhans.utils_tf import model_eval, model_argmax
import utils
import os
from os.path import expanduser
from cleverhans_tutorials.tutorial_models import make_basic_cnn_cifar10
from adversary_detector import detector

home = expanduser("~")+'/wangjingyi'
dataset = 'cifar10'
attack_type = 'jsma'
folder_name = dataset+'_'+attack_type

print('--- Dataset: ', dataset, 'attack type: ', attack_type)

image_rows = 32
image_cols = 32
channels = 3

# Restore the DNN model
sess = tf.Session()
# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, image_rows, image_cols, channels))
# y = tf.placeholder(tf.float32, shape=(None, 10))

# Define TF model graph
model = make_basic_cnn_cifar10()
preds = model(x)
print("Defined TensorFlow model graph.")

saver = tf.train.Saver()
saved_model_path = os.path.join(home + '/cleverhans/cleverhans_tutorials/'+dataset+'_'+attack_type+'/model', attack_type+'.model')
print('--- Restoring trained model from: ', saved_model_path)
saver.restore(sess, saved_model_path)


# Define a detector
ad = detector(0.0019, 1, image_rows, image_cols, 1, 2000, 0.05, 0.05, 0.0019*0.8)
print('--- Detector config: ', ad.print_config())

adv_image_dir = home+'/cleverhans/cleverhans_tutorials/'+dataset+'_adv_'+attack_type
print('--- Extracting adversary images from: ', adv_image_dir)
[adv_image_list, adv_image_files, real_labels, predicted_labels] = utils.get_data_mutation_test(adv_image_dir)
adv_count = 0
total_mutation_counts = []
label_change_mutation_counts = []

print('--- Evaluating adversarial inputs ---')

not_decided_images = 0
for i in range(len(adv_image_list)):
    print('- Running image ', i)
    # ori_img = np.expand_dims(adv_image_list[i], axis=2)
    ori_img = adv_image_list[i].astype('float32')
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



