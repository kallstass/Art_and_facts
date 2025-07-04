"""

"""

from core import models, image

from fastapi import FastAPI
import numpy as np

app = FastAPI()

class_dict = {0: 'Abstract_Expressionism',
 1: 'Action_painting',
 2: 'Analytical_Cubism',
 3: 'Art_Nouveau_Modern',
 4: 'Baroque',
 5: 'Color_Field_Painting',
 6: 'Contemporary_Realism',
 7: 'Cubism',
 8: 'Early_Renaissance',
 9: 'Expressionism',
 10: 'Fauvism',
 11: 'High_Renaissance',
 12: 'Impressionism',
 13: 'Mannerism_Late_Renaissance',
 14: 'Minimalism',
 15: 'Naive_Art_Primitivism',
 16: 'New_Realism',
 17: 'Northern_Renaissance',
 18: 'Pointillism',
 19: 'Pop_Art',
 20: 'Post_Impressionism',
 21: 'Realism',
 22: 'Rococo',
 23: 'Romanticism',
 24: 'Symbolism',
 25: 'Synthetic_Cubism',
 26: 'Ukiyo_e'}

@app.get("/")
async def root():
    loaded_model = models.load_model('model/RNN_batchsize512_32x32_full_dataset.keras')

    img_array = image.load_image('../../data/peinture_art/Abstract_Expressionism/aaron-siskind_acolman-1-1955.jpg')

    predictions = loaded_model.predict(img_array)

    class_top3 = [class_dict.get(i) for i in np.flip(np.argsort(predictions[0]))[:3]]
    response = [{"top3": class_top3}]
    return response
