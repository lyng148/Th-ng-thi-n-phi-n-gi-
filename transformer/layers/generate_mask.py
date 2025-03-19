import numpy as np
import tensorflow as tf

def generate_padding_mask(inp):
    """
    Tạo padding mask cho các token padding (giá trị 0) trong câu đầu vào.

    Args:
        inp (tensor): Câu đầu vào, shape: (..., seq_length)
                     Mỗi phần tử là một index trong từ điển

    Returns:
        tensor: Padding mask, shape: (..., 1, 1, seq_length)
               Giá trị 1 tại vị trí padding, 0 tại vị trí từ thực
    """
    result = tf.cast(inp == 0, dtype=tf.float32)[:, np.newaxis, np.newaxis, :]
    return result

def generate_look_ahead_mask(inp_len):
    """
    Tạo look-ahead mask để đảm bảo tính tuần tự khi dịch.
    Mỗi vị trí chỉ được nhìn thấy các từ phía trước nó.

    Args:
        inp_len (int): Độ dài của câu cần tạo mask

    Returns:
        tensor: Look-ahead mask, shape: (inp_len, inp_len)
               Ma trận tam giác trên với 1s, tam giác dưới với 0s
    """
    mask = 1 - tf.linalg.band_part(tf.ones((inp_len, inp_len)), -1, 0)
    return mask

def generate_mask(inp, targ):
    """
    Tạo tất cả các loại mask cần thiết cho Transformer.

    Args:
        inp (tensor): Câu nguồn (source sequence), shape: (..., src_len)
        targ (tensor): Câu đích (target sequence), shape: (..., targ_len)

    Returns:
        tuple: (encoder_padding_mask, decoder_padding_mask, decoder_look_ahead_mask)
            - encoder_padding_mask: Mask padding cho encoder
              Shape: (..., 1, 1, src_len)
            - decoder_padding_mask: Mask padding cho decoder khi attention với encoder
              Shape: (..., 1, 1, src_len)
            - decoder_look_ahead_mask: Kết hợp look-ahead và padding mask cho decoder
              Shape: (..., targ_len, targ_len)
    """
    encoder_padding_mask = generate_padding_mask(inp)
    decoder_padding_mask = generate_padding_mask(inp)
    decoder_look_ahead_mask = generate_look_ahead_mask(targ.shape[1])
    decoder_inp_padding_mask = generate_padding_mask(targ)
    decoder_look_ahead_mask = tf.maximum(decoder_look_ahead_mask, decoder_inp_padding_mask)

    return encoder_padding_mask, decoder_look_ahead_mask ,decoder_padding_mask

print(generate_look_ahead_mask(3))