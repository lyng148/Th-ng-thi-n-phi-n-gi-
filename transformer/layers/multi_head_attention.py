import tensorflow as tf
from tensorflow.keras.layers import Dense


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model = 512, h = 6):
        """
        Args:
            d_model (int): Số chiều của vector sau khi project q, k, v. Defaults to 512
            h (int): Số lượng head attention. Defaults to 6
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.wo = Dense(d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Args:
            q (tensor): Query tensor, shape: (..., q_length, d_k)
                       Câu truy vấn cần tìm attention
            k (tensor): Key tensor, shape: (..., k_length, d_k)
                       Các key để so sánh với query
            v (tensor): Value tensor, shape: (..., v_length, d_v)
                       Các value tương ứng với key
            mask (tensor, optional): Mask để che giấu một số từ. Defaults to None

        Returns:
            tuple: (out, attention_weights)
                - out: tensor, shape (..., q_length, d_v)
                  Kết quả attention: tổng có trọng số của các value
                - attention_weights: tensor, shape (..., q_length, k_length)
                  Ma trận trọng số attention giữa query và key
        """
        dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
        attention_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)

        if mask:
            attention_scores += (mask * -1e30)

        attention_weights =  tf.nn.softmax(attention_scores, axis=-1) 
        out = tf.matmul(attention_weights, v)

        return out, attention_weights

    def splitting_head(self, x):
        """
        Args:
            x (tensor): Input tensor (q/k/v), shape: (..., length, d_model)
                       Tensor cần chia thành nhiều head

        Returns:
            tensor: Tensor sau khi chia head
                   Shape: (..., h, length, d_model/h)
        """
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1] 
        d_model = tf.shape(x)[2] 
        
        hd_v = d_model // self.h
        x = tf.reshape(x, (batch_size, length, self.h, hd_v)) 
        xs = tf.transpose(x, [0, 2, 1, 3])
        
        return xs

    def call(self, q, k, v, mask=None):
        """
        Args:
            q (tensor): Query tensor, shape: (..., q_length, d_model)
                       Câu truy vấn cần tìm attention
            k (tensor): Key tensor, shape: (..., k_length, d_model)
                       Các key để so sánh với query
            v (tensor): Value tensor, shape: (..., v_length, d_model)
                       Các value tương ứng với key
            mask (tensor, optional): Mask để che giấu một số từ. Defaults to None

        Returns:
            tuple: (final, attention_weights)
                - final: tensor, shape (..., q_length, d_model)
                  Kết quả cuối cùng sau khi qua multi-head attention
                - attention_weights: tensor, shape (..., h, q_length, k_length)
                  Ma trận trọng số attention của mỗi head
        """
        batch_size = tf.shape(q)[0]
        qw = self.wq(q)
        kw = self.wk(k)
        vw = self.wv(v)

        heads_qw = self.splitting_head(qw)
        heads_kw = self.splitting_head(kw)
        heads_vw = self.splitting_head(vw)

        out, attention_weights = self.scaled_dot_product_attention(heads_qw, heads_kw, heads_vw)

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, tf.shape(qw)[1], self.d_model))
        final = self.wo(out)

        return final, attention_weights
