"""

"""

import tensorflow as tf

img_height = 32
img_width = 32


def load_image(path:str):
    new_img = tf.keras.utils.load_img(path, target_size=(img_height, img_width))

    img_array = tf.keras.utils.img_to_array(new_img)
    img_array = tf.expand_dims(img_array, 0)

    return img_array