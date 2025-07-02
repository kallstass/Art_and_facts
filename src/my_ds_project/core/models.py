"""
Machine learning models for the boilerplate.
"""
import tensorflow as tf
from tensorflow import keras


def load_model(path: str):
    """
    Load a model from the specified path.
    
    Args:
        path (str): The file path from which the model will be loaded.
        
    Returns:
        The loaded model.
    """
    loaded_model = tf.keras.models.load_model(path)
    return loaded_model
