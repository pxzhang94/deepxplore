import utils
import tensorflow as tf
from gen_mutation import MutationTest

# Get the adversary images as well as their labels
[image_list, real_labels, predicted_labels] = utils.get_data_mutation_test('/Users/jingyi/cleverhans-master/cleverhans_tutorials/adv_jsma')
img_rows = 28
img_cols = 28
step_size = 1
seed_number = 50
mutation_number = 10

# Get the trained model
print("Created TensorFlow session.")
sess = tf.Session()

print('Restore trained model ...')
new_saver = tf.train.import_meta_graph('/Users/jingyi/cleverhans-master/cleverhans_tutorials/MNIST_trained_model_jsma.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('/Users/jingyi/cleverhans-master/cleverhans_tutorials/'))

mutation_test = MutationTest(img_rows, img_cols, step_size, seed_number, mutation_number)
mutation_test.label_change_mutation_test(sess, image_list, predicted_labels)
