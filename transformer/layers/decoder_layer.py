import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.position_wise_feed_forward_network import ffn

class DecoderLayer(tf.keras.layers.Layer):
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
        super(DecoderLayer, self).__init__()
        self.masked_mtha = MultiHeadAttention(d_model, h)
        self.mtha = MultiHeadAttention(d_model, h)
        self.feed_forward = ffn(d_ff, d_model, activation)
        self.layernorm1 = LayerNormalization(epsilon=eps)
        self.layernorm2 = LayerNormalization(epsilon=eps)
        self.layernorm3 = LayerNormalization(epsilon=eps)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, q, encoder_out, is_train, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            q (tensor): Đầu vào của decoder, shape: (..., q_length, d_model)
                       Câu đang được dịch (target sequence)
            encoder_out (tensor): Đầu ra của encoder, shape: (..., src_length, d_model)
                                Biểu diễn của câu nguồn (source sequence)
            is_train (bool): Flag chỉ định đang trong quá trình training hay không
            look_ahead_mask (tensor, optional): Mask để đảm bảo tính tuần tự khi dịch
                                             Shape: (..., q_length, q_length)
            padding_mask (tensor, optional): Mask cho các padding token trong câu nguồn
                                          Shape: (..., q_length, src_length)

        Returns:
            tuple: (out, self_attn_weights, global_attn_weights)
                - out: tensor, shape (..., q_length, d_model)
                  Biểu diễn mới của câu đích sau khi qua decoder layer
                - self_attn_weights: tensor, shape (..., h, q_length, q_length)
                  Ma trận attention giữa các từ trong câu đích
                - global_attn_weights: tensor, shape (..., h, q_length, src_length)
                  Ma trận attention giữa câu đích và câu nguồn
        """
        k = v = encoder_out

        masked_mtha_out, self_attn_weights = self.masked_mtha(q, q, q, look_ahead_mask)
        q = self.layernorm1(q + self.dropout1(masked_mtha_out, training=is_train))

        mtha_out, global_attn_weights = self.mtha(q, k, v, padding_mask)
        q = self.layernorm2(q + self.dropout2(mtha_out, training=is_train))

        ffn_out = self.feed_forward(q)
        out = self.layernorm3(q + self.dropout3(ffn_out, training=is_train))

        return out, self_attn_weights, global_attn_weights