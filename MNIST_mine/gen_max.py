'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *
import time
import os

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
# parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
# parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)

args = parser.parse_args()

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(_, _), (x_test, _) = mnist.load_data()

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

# init coverage table
model_layer_dict1 = init_coverage_tables(model)

info = ""
info = info + "begin time: " + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + "\n"
print("begin time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# ==============================================================================================
# start gen inputs
for i in xrange(args.seeds):
    gen_img = np.expand_dims(random.choice(x_test), axis=0)
    orig_img = gen_img.copy()
    # first check if input already induces differences
    label1 = np.argmax(model.predict(gen_img)[0])

    update_coverage(gen_img, model, model_layer_dict1, args.threshold)
    info = info + "covered neurons percentage " + str(len(model_layer_dict1)) + " neurons " + str(neuron_covered(model_layer_dict1)[2]) + "\n"
    print(i, bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f'
          % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2]) + bcolors.ENDC)

    # if all label agrees
    orig_label = label1

    loss1_neuron = K.mean(model1.get_layer("before_softmax").output[..., label1])

    layer_output = loss1_neuron

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1_neuron, grads])

    # we run gradient ascent for 20 steps
    for iters in xrange(args.grad_iterations):
        loss_neuron1, grads_value = iterate([gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * args.step
        predictions1 = np.argmax(model.predict(gen_img)[0])

        if not predictions1 == orig_label:
            update_coverage(gen_img, model, model_layer_dict1, args.threshold)

            info = info + "covered neurons percentage " + str(len(model_layer_dict1)) + " neurons " + \
                  str(neuron_covered(model_layer_dict1)[2]) + "\n"
            print(i, bcolors.OKBLUE + 'covered neurons percentage %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2]) + bcolors.ENDC)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            store_path = './generated_inputs/model_' + str(args.target_model) + "/" + args.transformation + "/" + str(i)
            isExists = os.path.exists(store_path)
            if not isExists:
                os.makedirs(store_path)

            # save the result to disk
            imsave(store_path + "/" + str(predictions1) + "_" + str(orig_label) + '.png',
                   gen_img_deprocessed)
            imsave(store_path + "/" + str(orig_label) + '_orig.png',
                   orig_img_deprocessed)
            break

info = info + "end time: " + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
f = open('./generated_inputs/model_' + str(args.target_model) + "/" + args.transformation + "/coverage.txt", 'w')
f.write(info)
f.close()
print("end time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
