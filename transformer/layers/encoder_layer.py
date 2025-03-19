import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.position_wise_feed_forward_network import ffn

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, h, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
        """
        Args:
            h (int): Số lượng head trong Multi-Head Attention
            d_model (int): Số chiều của vector embedding và hidden states
            d_ff (int): Số chiều của Feed Forward Network
            activation (str): Hàm kích hoạt trong Feed Forward Network
            dropout_rate (float, optional): Tỉ lệ dropout. Defaults to 0.1
            eps (float, optional): Epsilon cho Layer Normalization. Defaults to 0.1
        """
        super(EncoderLayer, self).__init__()
        self.mtha = MultiHeadAttention(d_model, h)
        self.feed_forward = ffn(d_ff, d_model, activation)
        self.layernorm1 = LayerNormalization(epsilon=eps)
        self.layernorm2 = LayerNormalization(epsilon=eps)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, is_train, mask=None):
        """
        Args:
            x (tensor): Input tensor, shape: (..., q_length, d_model)
                       Đầu vào cần được xử lý qua encoder layer
            is_train (bool): Flag chỉ định đang trong quá trình training hay không
            mask (tensor, optional): Mask để che giấu một số từ. Defaults to None
                                   Shape: (..., q_length, q_length)

        Returns:
            tensor: Biểu diễn mới của input sau khi qua self-attention và feed forward
                   Shape: (..., q_length, d_model)
        """
        q = x

        mtha_out, _ = self.mtha(q, q, q, mask)

        x = self.layernorm1(q + self.dropout1(mtha_out, training=is_train))

        ffn_out = self.feed_forward(x)

        out = self.layernorm2(x + self.dropout2(ffn_out, training=is_train))

        return out