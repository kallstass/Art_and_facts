"""

"""

from src.my_ds_project.core import models, image

import tensorflow as tf



if __name__ == '__main__':
    loaded_model = models.load_model('model/RNN_batchsize512_32x32_full_dataset.keras')

    img_array = image.load_image('../../data/peinture_art/Abstract_Expressionism/aaron-siskind_acolman-1-1955.jpg')

    predictions = loaded_model.predict(img_array)
    print(predictions[0])
