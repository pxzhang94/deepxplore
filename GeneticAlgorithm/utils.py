import numpy as np
import random


def initialization(img, num, stepsize):
    img_list = []

    for i in range(num):
        method = random.randint(1, 3)
        trans_matrix = generate_value(len(img), len(img[0]))
        rect_shape = (random.randint(1, 10), random.randint(1, 10))
        start_point = (
            random.randint(0, np.asarray(img).shape[1] - rect_shape[0]),
            random.randint(0, np.asarray(img).shape[2] - rect_shape[1]))

        if method == 1:
            transformation = constraint_light(trans_matrix)
        elif method == 2:
            transformation = constraint_occl(trans_matrix, start_point, rect_shape)
        elif method == 3:
            transformation = constraint_black(trans_matrix, start_point, rect_shape)
        img_list.append(img + transformation * stepsize)
    return img_list

#it is easy to choose each gene randomly between parents
def crossover(img1, img2):
    gen_img1 = np.zeros_like(img1)
    gen_img2 = np.zeros_like(img1)
    for i in range(len(img1)):
        for j in range(len(img1[i])):
            for k in range(len(img1[i][j])):
                for l in range(len(img1[i][j][k])):
                    choose = random.randint(1, 2)
                    if choose == 1:
                        gen_img1[i][j][k][l] = img1[i][j][k][l]
                        gen_img2[i][j][k][l] = img2[i][j][k][l]
                    elif choose == 2:
                        gen_img1[i][j][k][l] = img2[i][j][k][l]
                        gen_img2[i][j][k][l] = img1[i][j][k][l]
    return gen_img1, gen_img2

def mutatation(img, stepsize):
    method = random.randint(1, 3)
    trans_matrix = generate_value(len(img), len(img[0]))
    rect_shape = (random.randint(1, 5), random.randint(1, 5)) #mutation area is smaller than initialization
    start_point = (
        random.randint(0, np.asarray(img).shape[1] - rect_shape[0]),
        random.randint(0, np.asarray(img).shape[2] - rect_shape[1]))

    if method == 1:
        transformation = constraint_light(trans_matrix)
    elif method == 2:
        transformation = constraint_occl(trans_matrix, start_point, rect_shape)
    elif method == 3:
        transformation = constraint_black(trans_matrix, start_point, rect_shape)
    return img + transformation * stepsize

def fitness(img1, img2, result1, result2):
    x_differential = distance_x(img1, img2)
    y_differential = distance_y(result1, result2)
    return y_differential / x_differential

def generate_value(row, col):
    matrix = random.random(size = (row, col))
    for i in range(row):
        for j in range(col):
            matrix[i][j] = (matrix[i][j] - 0.5) * 4
    return matrix

# Because all the pic from one original pic(only pixel changed), it issuitable to use euclidean metric
def distance_x(img1, img2):
    sum = 0
    for i in range(len(img1)):
        for j in range(len(img1[i])):
            for k in range(len(img1[i][j])):
                for l in range(len(img1[i][j][k])):
                    sum += (img1[i][j][k][l] - img2[i][j][k][l]) ** 2
    return sum ** 0.5

def distance_y(result1, result2):
    sum = 0
    for i in range(len(result1)):
        sum += (result1[i] - result2[j]) ** 2
    return sum ** 0.5

def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads

def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = 500 * np.mean(gradients)
    return grad_mean * new_grads

def constraint_black(gradients, start_point, rect_shape):
    # start_point = (
    #     random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads