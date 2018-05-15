import tensorflow as tf
import  numpy as np
import os

img_width = 208
img_height = 208

'''
获得图片以及对应的标签
Input: 
    图片的路径 
Returns:
    图片列表和图片标签列表
'''
def get_files(file_dir):
    cats = []
    lable_cats = []
    dogs = []
    lable_dogs = []

    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            lable_cats.append(0)
        else:
            dogs.append(file_dir + file)
            lable_dogs.append(1)
    print('There are %d cats and %d dogs' %(len(cats),len(dogs)))

    image_list = np.hstack((cats, dogs))
    lable_list = np.hstack((lable_cats,lable_dogs))

    temp = np.array([image_list,lable_list])
    # print(temp)
    temp = temp.transpose()
    # print(temp)
    np.random.shuffle(temp)
    # print(temp)

    image_list = list(temp[:,0])
    # print(image_list)
    lable_list = list(temp[:,1])
    # print(lable_list)
    # Change str to int
    lable_list = [int(i) for i in lable_list]
    # print(lable_list)

    return image_list, lable_list

'''
获得批次
对图片一批一批地处理
'''
def get_batch(image_list, label, image_W, image_H, batch_size, capacity):

    image = tf.cast(image_list, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成一个队列
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 此处可以添加一些图像处理功能
    # image  = tf.image.

    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    # image = tf.image.resize_images(image, [image_W, image_H])
    # image = tf.image.resize_nearest_neighbor(image, [image_W,image_H], align_corners=False, name=None)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                              num_threads=64, capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


# get_files('data/train/')
# 测试
#
# import matplotlib.pyplot as plt
# BATCH_SIZE = 10
# CAPACITY = 512
# IMG_W = 500
# IMG_H = 500
# train_dir = 'data/train/'
#
# image_list, label_list = get_files(train_dir)
# image_batch, lable_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#     i = 0
#     # 用来监控队列的状态
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     try:
#         while not coord.should_stop() and i < 1:
#             img, label = sess.run([image_batch, lable_batch])
#
#             for j in np.arange(BATCH_SIZE):
#                 print('lable: %d' %label[j])
#                 # print(img[j,:,:,:])
#                 plt.imshow(img[j,:,:,:])
#                 plt.show()
#             i += 1
#     except tf.errors.OutOfRangeError:
#         print('Done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)

