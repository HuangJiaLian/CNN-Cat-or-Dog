import os
import numpy as np
import tensorflow as tf
import model
from PIL import Image
import matplotlib.pyplot as plt

def get_test_file(file_dir):
    pics = []
    for file in os.listdir(file_dir):
        name = file
        pics.append(file_dir + file)

    print('There are %d ' %(len(pics)))
    image_list = np.hstack((pics))
    np.random.shuffle(image_list)
    print(image_list)

    return image_list


def get_one_image(test):
   '''Randomly pick one image from test data
   Return: ndarray
   '''
   global imageForShow
   n = len(test)
   ind = np.random.randint(0, n)
   img_dir = test[ind]

   image = Image.open(img_dir)
   imageForShow = image
   image = image.resize([208, 208])
   image = np.array(image)
   return image


def evaluate_one_image():
   DIR_PRE = os.getcwd() + '/'
   test_dir = DIR_PRE + 'data/test1/'
   logs_train_dir = DIR_PRE + 'logs/train/'



   with tf.Graph().as_default():

       test = get_test_file(test_dir)
       image_array = get_one_image(test)

       BATCH_SIZE = 1
       N_CLASSES = 2

       # image = tf.cast(image_array, tf.float32)
       # image = tf.image.per_image_standardization(image)
       # image = tf.reshape(image, [1, 208, 208, 3])
       image = tf.reshape(tf.image.per_image_standardization(tf.cast(image_array, tf.float32)), [1, 208, 208, 3])
       # logit = model.inference(image, BATCH_SIZE, N_CLASSES)
       # logit = tf.nn.softmax(logit)
       logit = tf.nn.softmax(model.inference(image, BATCH_SIZE, N_CLASSES))
       x = tf.placeholder(tf.float32, shape=[208, 208, 3])

       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           while True:
               image_array = get_one_image(test)
               # print(image_array)
               prediction = sess.run(logit, feed_dict={x: image_array})
               max_index = np.argmax(prediction)
               if max_index==0:
                   print('This is a cat with possibility %.6f' %prediction[:, 0])
                   plt.title("This is a cat")
               else:
                   print('This is a dog with possibility %.6f' %prediction[:, 1])
                   plt.title("This is a dog")
               plt.imshow(imageForShow)
               plt.show()

evaluate_one_image()

