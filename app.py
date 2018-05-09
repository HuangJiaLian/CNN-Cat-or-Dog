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


def get_one_image(train):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)
   plt.imshow(image)
   plt.show()
   image = image.resize([208, 208])
   image = np.array(image)
   return image


def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''

   # you need to change the directories to yours.
   DIR_PRE = '/home/dell01/github/CNN-Cat-or-Dog/'
   test_dir = DIR_PRE + 'data/test1/'
   test = get_test_file(test_dir)
   image_array = get_one_image(test)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 2

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 208, 208, 3])
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)
       logit = tf.nn.softmax(logit)
       x = tf.placeholder(tf.float32, shape=[208, 208, 3])

       # you need to change the directories to yours.
       DIR_PRE = '/home/dell01/github/CNN-Cat-or-Dog/'
       logs_train_dir = DIR_PRE + 'logs/train/'

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

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('This is a cat with possibility %.6f' %prediction[:, 0])
           else:
               print('This is a dog with possibility %.6f' %prediction[:, 1])


evaluate_one_image()