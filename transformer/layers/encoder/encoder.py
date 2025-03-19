import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout
from transformer.layers.encoder_layer import EncoderLayer
from transformer.layers.generate_position import generate_positional_encoding


class Encoder(tf.keras.layers.Layer):
    """
    Encoder trong mô hình Transformer.

    Encoder bao gồm các thành phần chính:
    1. Word Embedding: Chuyển đổi các từ thành vector embedding
    2. Positional Encoding: Thêm thông tin về vị trí của từng từ trong câu
    3. N lớp Encoder Layer: Mỗi lớp có Multi-Head Attention và Feed Forward Network

    Attributes:
        n (int): Số lượng Encoder Layer (số lớp xếp chồng)
        h (int): Số lượng head trong Multi-Head Attention
        vocab_size (int): Kích thước từ điển (số lượng từ khác nhau)
        d_model (int): Số chiều của vector embedding và hidden states
        d_ff (int): Số chiều của Feed Forward Network
        activation (str): Hàm kích hoạt trong Feed Forward Network
        dropout_rate (float): Tỉ lệ dropout để chống overfitting
        eps (float): Epsilon dùng trong Layer Normalization
    """
    def __init__(self, n, h, vocab_size, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
        """
        Khởi tạo Encoder.

        Args:
            n (int): Số lượng Encoder Layer
            h (int): Số lượng head trong Multi-Head Attention
            vocab_size (int): Kích thước từ điển
            d_model (int): Số chiều của vector embedding
            d_ff (int): Số chiều của Feed Forward Network
            activation (str): Hàm kích hoạt (e.g., 'relu')
            dropout_rate (float, optional): Tỉ lệ dropout. Defaults to 0.1
            eps (float, optional): Epsilon cho Layer Normalization. Defaults to 0.1
        """
        super(Encoder, self).__init__()
        self.n = n
        self.d_model = d_model
        self.encoder_layers = [EncoderLayer(h, d_model, d_ff, activation, dropout_rate, eps) for _ in range(n)]
        self.word_embedding = Embedding(vocab_size, output_dim=d_model)
        self.dropout = Dropout(dropout_rate)

    def call(self, q, is_train, mask=None):
        """
        Thực hiện quá trình encoding cho câu đầu vào.

        Các bước xử lý:
        1. Chuyển đổi từng từ thành vector embedding
        2. Scale embedding bằng căn bậc hai của d_model
        3. Thêm positional encoding
        4. Áp dụng dropout
        5. Đưa qua các encoder layer

        Args:
            q (tensor): Query - Câu đầu vào dưới dạng chuỗi các chỉ số trong từ điển
                       Ví dụ: câu "I love you" -> [5, 20, 15] với 5,20,15 là chỉ số trong từ điển
                       Shape: (..., q_length) với q_length là độ dài câu
            is_train (bool): Flag chỉ định đang trong quá trình training hay không
            mask (tensor, optional): Mask để che giấu một số từ. Defaults to None
                                  Shape: (..., q_length, q_length)

        Returns:
            tensor: Biểu diễn mới của câu sau khi qua encoder
                   Shape: (..., q_length, d_model)
                   Mỗi từ ban đầu được biểu diễn bởi vector d_model chiều
        """
        q_length = q.shape[1]  # Độ dài của câu đầu vào
        
        # Chuyển đổi từ thành vector embedding
        # Mỗi chỉ số trong q được chuyển thành vector d_model chiều
        # Shape: (..., q_length, d_model)
        embedded_q = self.word_embedding(q)

        # Scale embedding để tránh gradient quá lớn
        embedded_q *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Thêm positional encoding để mô hình biết vị trí của từng từ
        # Shape: (1, q_length, d_model)
        positional_encoding = generate_positional_encoding(q_length, self.d_model)

        # Áp dụng dropout và cộng với positional encoding
        encoder_out = self.dropout(embedded_q + positional_encoding, training=is_train)

        # Đưa qua từng encoder layer
        for encoder_layer in self.encoder_layers:
            encoder_out = encoder_layer(encoder_out, is_train, mask)
        
        return encoder_out
