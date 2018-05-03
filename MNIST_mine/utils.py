import random
from collections import defaultdict

import numpy as np
import os
from keras import backend as K
from keras.models import Model
from scipy import ndimage

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255)#.astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads

def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


# def init_coverage_tables(model1, model2, model3):
#     model_layer_dict1 = defaultdict(bool)
#     model_layer_dict2 = defaultdict(bool)
#     model_layer_dict3 = defaultdict(bool)
#     init_dict(model1, model_layer_dict1)
#     init_dict(model2, model_layer_dict2)
#     init_dict(model3, model_layer_dict3)
#     return model_layer_dict1, model_layer_dict2, model_layer_dict3
def init_coverage_tables(model1):
    model_layer_dict1 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    return model_layer_dict1


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False

def c_occl(gradients, start_point, rect_shape):
    gradients = np.asarray(gradients)
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads

def c_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads

def c_black(gradients, start_point, rect_shape):
    # start_point = (
    #     random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    gradients = np.asarray(gradients)
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads

def generate_value(row, col):
    matrix = []
    for i in range(row):
        line = []
        for j in range(col):
            pixel = []
            for k in range(1):
                div = random.randint(1, 20)#5,100
                pixel.append((random.random() - 0.5) / div)
            line.append(pixel)
        matrix.append(line)
    return [matrix]

# # generate for RGB
# def generate_value(row, col):
#     matrix = []
#     for i in range(row):
#         line = []
#         for j in range(col):
#             pixel = []
#             for k in range(3):
#                 div = random.randint(1, 20)  # 5,100
#                 pixel.append((random.random() - 0.5) * 4 / div)
#             line.append(pixel)
#         matrix.append(line)
#     # matrix = random.random(size = (row, col))
#     # for i in range(row):
#     #     for j in range(col):
#     #         matrix[i][j] = (matrix[i][j] - 0.5) * 4
#     return [matrix]


def get_data_mutation_test(file_path):
    '''
    :param file_path: the file path for the adversary images
    :return: the formatted data for mutation test, the actual label of the images, and the predicted label of the images
    '''

    image_list = []
    real_labels = []
    predicted_labels = []
    image_files =[]
    for img_file in os.listdir(file_path):
        if img_file.endswith('.png'):
            print('Reading image: ', img_file)
            img_file_split = img_file.split('_')
            real_labels.append(img_file_split[-3])
            predicted_labels.append(img_file_split[-2])
            current_img = ndimage.imread(file_path + os.sep + img_file)
            print(current_img.shape)
            image_list.append(current_img)
            image_files.append(img_file)
    print('Real labels: ', real_labels)
    print('Predicted labels: ', predicted_labels)
    return image_list, image_files, real_labels, predicted_labels

# get_data_mutation_test('/Users/jingyi/cleverhans-master/cleverhans_tutorials/adv_jsma')