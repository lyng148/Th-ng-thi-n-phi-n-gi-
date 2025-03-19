import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout
from transformer.layers.decoder_layer import DecoderLayer
from transformer.layers.generate_position import generate_positional_encoding


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n, h, vocab_size, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
        """
        Args:
            n (int): Số lượng decoder layer
            h (int): Số lượng head trong Multi-Head Attention
            vocab_size (int): Kích thước từ điển đầu ra
            d_model (int): Số chiều của vector embedding và hidden states
            d_ff (int): Số chiều của Feed Forward Network
            activation (str): Hàm kích hoạt trong Feed Forward Network
            dropout_rate (float, optional): Tỉ lệ dropout. Defaults to 0.1
            eps (float, optional): Epsilon cho Layer Normalization. Defaults to 0.1
        """
        super(Decoder, self).__init__()
        self.n = n
        self.d_model = d_model
        self.decoder_layers = [DecoderLayer(h, d_model, d_ff, activation, dropout_rate, eps) for _ in range(n)]
        self.word_embedding = Embedding(vocab_size, output_dim=d_model)
        self.dropout = Dropout(dropout_rate)

    def call(self, q, encoder_out, is_train, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            q (tensor): Câu đích (target sentence), shape: (..., q_length)
                       Chuỗi các index trong từ điển đầu ra
            encoder_out (tensor): Đầu ra của encoder, shape: (..., k_length, d_model)
                                Biểu diễn của câu nguồn sau khi qua encoder
            is_train (bool): Flag chỉ định đang trong quá trình training hay không
            look_ahead_mask (tensor, optional): Mask để đảm bảo tính tuần tự khi dịch
            padding_mask (tensor, optional): Mask cho các padding token trong câu nguồn

        Returns:
            tuple: (decoder_out, attentionWeights)
                - decoder_out: tensor, shape (..., q_length, d_model)
                  Biểu diễn mới của câu đích sau khi qua decoder
                - attentionWeights: dict
                  + decoder_layer_{i}_self_attn_weights: Ma trận attention giữa các từ trong câu đích
                    Shape: (..., q_length, q_length)
                  + decoder_layer_{i}_global_attn_weights: Ma trận attention giữa câu đích và nguồn
                    Shape: (..., q_length, k_length)
        """
        q_length = q.shape[1]
        
        # embedded_q shape: (..., q_length, d_model)
        # TODO: Normalize embedded_q
        embedded_q = self.word_embedding(q)

        # positional_encoding shape: (1, q_length, d_model)
        positional_encoding = generate_positional_encoding(q_length, self.d_model)

        decoder_out = self.dropout(embedded_q + positional_encoding, training=is_train)

        attention_weights = {}

        # Do Attention
        # decoder_out shape: (..., q_length, d_model)
        for i, decoder_layer in enumerate(self.decoder_layers):
            decoder_out, self_attn_weights, global_attn_weights = decoder_layer(decoder_out, encoder_out, is_train, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer_{}_self_attn_weights'.format(i)] = self_attn_weights
            attention_weights['decoder_layer_{}_global_attn_weights'.format(i)] = global_attn_weights
        
        return decoder_out, attention_weights
