import numpy as np
import tensorflow as tf


def createAngleRates(d_model):
    """
    Creates angle rates for positional encoding calculation in the Transformer model.
    
    This function generates the angle rates used in the positional encoding formula:
    angle_rates = 1 / (10000 ** (2i/d_model)) where i is the dimension.
    
    Args:
        d_model (int): so chieu cua embedding vector cua model
        
    Returns:
        numpy.ndarray: A 1xd_model matrix containing the angle rates
    """
    angles = np.arange(d_model)
    angles[1::2] = angles[0::2]
    angles = 1 / (10000 ** (angles / d_model))
    angles = np.expand_dims(angles, axis=0)
    return angles

def generate_positional_encoding(pos, d_model):
    """
    Generates positional encodings for the Transformer model.
    
    This function implements the positional encoding as described in the paper
    "Attention Is All You Need". The encoding uses sine and cosine functions of different
    frequencies to create unique position embeddings:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        pos (int): Maximum sequence length for which to generate position encodings
        d_model (int): The dimension of the model's embedding vectors
        
    Returns:
        tensorflow.Tensor: A tensor of shape (1, pos, d_model) containing the positional
                         encodings for all positions up to pos
    """
    angles = createAngleRates(d_model)
    pos = np.expand_dims(np.arange(pos), axis=1)
    pos_angles = pos.dot(angles)
    pos_angles[:, 0::2] = np.sin(pos_angles[:, 0::2])
    pos_angles[:, 1::2] = np.cos(pos_angles[:, 1::2])
    pos_angles = np.expand_dims(pos_angles, axis=0)
  
    return tf.cast(pos_angles, dtype=tf.float32)

# unit test for 2 function
def test_createAngleRates():
    angles = createAngleRates(5)
    assert angles.shape == (1, 5)
    assert np.allclose(angles, np.array([[1, 1, 1, 1, 1]]))

def test_generate_positional_encoding():
    pos_angles = generate_positional_encoding(5, 5)
    assert pos_angles.shape == (1, 5, 5)
    assert np.allclose(pos_angles, np.array([[[1, 1, 1, 1, 1]]]))

test_createAngleRates()