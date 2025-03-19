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
    angles = np.arange(d_model)                 # tao array tu 0 den d_model - 1 (0, 1, ..., d_model - 1)
    angles[1::2] = angles[0::2]                 # copy value tu o chan sang o le -> [0, 0, 2, 2, ...]
    angles = 1 / (10000 ** (angles / d_model))  # tinh angle rate
    angles = np.expand_dims(angles, axis=0)     # vi sao phai them chieu ? dee ty nua matrix pos co shape (seq_length, 1), nhan voi matrix pos co shape (1, d_model)
    return angles                               # return matrix co shape (1, d_model)
                                            # seq_length chinh la pos
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
    """                                                 # no chi khac nhau o pos thoi, nen minh tinh chung roi nhan voi pos sau
    angles = createAngleRates(d_model)                  # return angle_matrix co shape (1, d_model) 
    pos = np.expand_dims(np.arange(pos), axis=1)        # bien pos thanh ma tran [0, 1, ..., pos - 1], sau khi expand  dims co shape = (pos, 1)
    pos_angles = pos.dot(angles)                        # tinh pos_angles co shape = (pos, d_model) CT: pos / 10000^(2i/d_model)
    pos_angles[:, 0::2] = np.sin(pos_angles[:, 0::2])   # tai tat ca cac hang cua pos_angles, tuong ung voi tung tu trong cau, tai cac o
                                                        # co vi tri chan, su dung ham sin
    pos_angles[:, 1::2] = np.cos(pos_angles[:, 1::2])   # tuong tu su dung cos voi o le
    pos_angles = np.expand_dims(pos_angles, axis=0)     # bien pos_angle thanh matrix co shape = (1, pos, d_model)
                                                         
    return tf.cast(pos_angles, dtype=tf.float32)
