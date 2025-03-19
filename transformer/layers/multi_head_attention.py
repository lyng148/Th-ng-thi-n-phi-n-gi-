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
        # de nhan duoc q * k.T, d_k cua q va k phai bang nhau
        # sau khi nhan, duoc ma tran co shape = (..., q_length, k_length)
        # de nhan duoc q* k.T voi v, v_length va k_length phai bang nhau
        dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)       # La value dk tu shpe cua k tensor va cast sang 'float32'
        attention_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk) # attention_score = q * k.T / sqrt(dk)
        if mask:
            attention_scores += (mask * -1e30)                # apply mask

        attention_weights =  tf.nn.softmax(attention_scores, axis=-1) # lay softmax theo chieu k_length
        out = tf.matmul(attention_weights, v) # out = attention_weights * v

        return out, attention_weights # out co shape (..., q_length, d_v)

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
                                   # h: so luong head
        hd_v = d_model // self.h   # hd_v: depth of each head
        x = tf.reshape(x, (batch_size, length, self.h, hd_v)) 
        xs = tf.transpose(x, [0, 2, 1, 3]) # xs co shape = (batch_size, h, length, hd_v)
        
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
        qw = self.wq(q)             # q co shape = (batch_size, q_length, d_model)
        kw = self.wk(k)             # k co shape = (batch_size, k_length, d_model)
        vw = self.wv(v)             # v co shape = (batch_size, v_length, d_model)
                                    # sau khi dua qua Dense(d_model)
        heads_qw = self.splitting_head(qw)   # heads_qw co shape = (batch_size, h, q_length, hd_v)
        heads_kw = self.splitting_head(kw)   # heads_kw co shape = (batch_size, h, k_length, hd_v)
        heads_vw = self.splitting_head(vw)   # heads_vw co shape = (batch_size, h, v_length, hd_v)

        out, attention_weights = self.scaled_dot_product_attention(heads_qw, heads_kw, heads_vw) # out co shape = (batch_size, h, q_length, hd_v)
    
        out = tf.transpose(out, [0, 2, 1, 3]) # transpose thanh shape = (batch_size, q_length, h, hd_v)
        out = tf.reshape(out, (batch_size, tf.shape(qw)[1], self.d_model)) # reshape lai thanh = (batch_size, q_length, d_model) voi d_model = h * hd_v
                                                                           # buoc nay dung de concat cac head lai do
        final = self.wo(out) # final co shape = (batch_size, q_length, d_model)

        return final, attention_weights
