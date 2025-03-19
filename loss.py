import tensorflow as tf
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    """
    Tính hàm mất mát cho mô hình Transformer, bỏ qua các padding token.
    
    Sử dụng SparseCategoricalCrossentropy và mask để chỉ tính loss trên các từ thực sự,
    không tính loss trên các padding token (giá trị 0).

    Args:
        real (tensor): Nhãn thực tế, shape (batch_size, seq_length)
        pred (tensor): Dự đoán của mô hình, shape (batch_size, seq_length, vocab_size)
        
    Returns:
        float: Giá trị loss trung bình trên các từ không phải padding
    """
    mask = tf.math.logical_not(real == 0)  # True cho các từ thực, False cho padding
    loss = loss_object(real, pred)  # Tính cross entropy loss
    
    mask = tf.cast(mask, dtype=loss.dtype)  # Chuyển mask sang cùng kiểu dữ liệu với loss
    loss = loss * mask  # Chỉ giữ lại loss của các từ thực
    
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)  # Lấy trung bình loss trên các từ thực