from transformer.layers.encoder import Encoder
from transformer.layers.decoder import Decoder
from tensorflow.keras.layers import Dense
from transformer.layers.generate_mask import generate_mask
import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

def cal_acc(real, pred):
    """
    Tính độ chính xác của dự đoán, bỏ qua các padding token.

    Args:
        real (tensor): Nhãn thực tế, shape (..., seq_length)
        pred (tensor): Dự đoán của mô hình, shape (..., seq_length, vocab_size)

    Returns:
        float: Độ chính xác trung bình trên các từ không phải padding
    """
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

class Transformer(tf.keras.models.Model):
    """
    Mô hình Transformer cho bài toán dịch máy.

    Args:
        n (int): Số lượng encoder và decoder layers
        h (int): Số lượng attention heads
        inp_vocab_size (int): Kích thước từ điển ngôn ngữ nguồn
        targ_vocab_size (int): Kích thước từ điển ngôn ngữ đích
        d_model (int): Số chiều của vector embedding và các layer trung gian
        d_ff (int): Số chiều của feed-forward network
        activation (str): Hàm kích hoạt cho feed-forward network
        dropout_rate (float, optional): Tỉ lệ dropout. Defaults to 0.1
        eps (float, optional): Epsilon cho layer normalization. Defaults to 0.1
    """
    def __init__(self, n, h, inp_vocab_size, targ_vocab_size, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n, h, inp_vocab_size, d_model, d_ff, activation, dropout_rate, eps)
        self.decoder = Decoder(n, h, targ_vocab_size, d_model, d_ff, activation, dropout_rate, eps)
        self.ffn = Dense(targ_vocab_size)

    def call(self, encoder_in, decoder_in, is_train, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask):
        """
        Thực hiện forward pass qua mô hình Transformer.

        Args:
            encoder_in (tensor): Input cho encoder, shape (batch_size, inp_seq_len)
            decoder_in (tensor): Input cho decoder, shape (batch_size, tar_seq_len)
            is_train (bool): True nếu đang trong quá trình training
            encoder_padding_mask (tensor): Mask cho encoder padding
            decoder_look_ahead_mask (tensor): Mask ngăn decoder nhìn thấy các token tương lai
            decoder_padding_mask (tensor): Mask cho decoder padding

        Returns:
            tensor: Output cuối cùng, shape (batch_size, tar_seq_len, target_vocab_size)
        """
        encoder_out = self.encoder(encoder_in, is_train, encoder_padding_mask)
        decoder_out, attention_weights = self.decoder(decoder_in, encoder_out, is_train, decoder_look_ahead_mask, decoder_padding_mask)
        ffn_out = self.ffn(decoder_out)
        return ffn_out