import argparse
from scipy.misc import imsave
from utils import *
import random
import numpy as np
from driving_models import *
import os

def mutation_matrix():
    method = random.randint(1, 3)
    trans_matrix = generate_value(img_rows, img_cols)
    rect_shape = (random.randint(1, 5), random.randint(1, 5))
    start_point = (
        random.randint(0, img_rows - rect_shape[0]),
        random.randint(0, img_cols - rect_shape[1]))

    if method == 1:
        transformation = c_light(trans_matrix)
    elif method == 2:
        transformation = c_occl(trans_matrix, start_point, rect_shape)
    elif method == 3:
        transformation = c_black(trans_matrix, start_point, rect_shape)

    return transformation

parser = argparse.ArgumentParser(
    description='Main function for mutation algorithm for input generation in Driving dataset')
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('mu_number', help="number of mutation", type=int)
args = parser.parse_args()
stepsize = args.step

# input image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model = Dave_orig(input_tensor=input_tensor, load_weights=True)

img_paths = image.list_pictures('./testing/center', ext='jpg')

mutation = []
for i in xrange(args.mu_number):
    mutation.append(mutation_matrix())

diverged = {}
for i in range(args.seeds):
    img_path = random.choice(img_paths)
    ori_img = preprocess_image(img_path)
    angle_ori = model.predict(ori_img)[0]
    orig_img_deprocessed = draw_arrow1(deprocess_image(ori_img), angle_ori)

    # save the result to disk
    store_path = './mutation_test/' + img_path.split("/")[-1][:-4]
    isExists = os.path.exists(store_path)
    if not isExists:
        os.makedirs(store_path)

    imsave(store_path + "/" + str(angle_ori) + '_orig.png', orig_img_deprocessed)

    count = 0
    for j in range(args.mu_number):
        img = ori_img.copy()
        mu_img = img + mutation[j] * stepsize
        angle_mu = model.predict(mu_img)[0]

        mu_img_deprocessed = draw_arrow2(deprocess_image(mu_img), angle_ori, angle_mu)
        imsave(store_path + "/" + str(angle_mu) + "_" + str(j) + '.png', mu_img_deprocessed)

        if angle_diverged(angle_mu, angle_ori):
            count += 1

    diverged[img_path.split("/")[-1][:-4]] = count

    path = './mutation_test/' + img_path.split("/")[-1][:-4] + "/" + str(count)
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

print(diverged)