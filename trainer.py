from transformer.layers.generate_mask import generate_mask
import tensorflow as tf

class Trainer:
    def __init__(self, model, optimizer, epochs, checkpoint_folder):
        """
        Args:
            model: Model Transformer cần huấn luyện
            optimizer: Thuật toán tối ưu (optimizer) để cập nhật trọng số
            epochs (int): Số epoch huấn luyện
            checkpoint_folder (str): Đường dẫn thư mục lưu checkpoint
        """
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_folder, max_to_keep=3)

    def cal_acc(self, real, pred):
        """
        Tính độ chính xác của dự đoán, bỏ qua các padding token.

        Args:
            real (tensor): Nhãn thực tế, shape: (..., seq_length)
            pred (tensor): Dự đoán của mô hình, shape: (..., seq_length, vocab_size)

        Returns:
            float: Độ chính xác trung bình trên các từ không phải padding
        """
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))
        mask = tf.math.logical_not(real == 0)
        accuracies = tf.math.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def loss_function(self, real, pred):
        """
        Tính hàm mất mát cross entropy, bỏ qua các padding token.

        Args:
            real (tensor): Nhãn thực tế, shape: (..., seq_length)
            pred (tensor): Dự đoán của mô hình, shape: (..., seq_length, vocab_size)

        Returns:
            float: Giá trị hàm mất mát trung bình trên các từ không phải padding
        """
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = cross_entropy(real, pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def train_step(self, inp, tar):
        """
        Thực hiện một bước huấn luyện trên một batch dữ liệu.

        Args:
            inp (tensor): Câu nguồn (source sentence), shape: (batch_size, src_len)
            tar (tensor): Câu đích (target sentence), shape: (batch_size, targ_len)

        Returns:
            tuple: (loss, accuracy)
                - loss (float): Giá trị hàm mất mát trên batch
                - accuracy (float): Độ chính xác trên batch
        """
        # TODO: Update document
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        encoder_padding_mask, decoder_look_ahead_mask ,decoder_padding_mask = generate_mask(inp, tar_inp)

        with tf.GradientTape() as tape:
            preds = self.model(inp, tar_inp, True, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask)
            d_loss = self.loss_function(tar_real, preds)

        # Compute gradients
        grads = tape.gradient(d_loss, self.model.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Compute metrics
        self.train_loss.update_state(d_loss)
        self.train_accuracy.update_state(self.cal_acc(tar_real, preds))

        # return {"loss": self.train_loss.result(), "acc": self.train_accuracy.result()}

    def fit(self, data):
        """
        Huấn luyện mô hình trên dữ liệu.

        Args:
            data: Dữ liệu huấn luyện, có thể là tf.data.Dataset hoặc iterable
        """
        print('=============Training Progress================')
        print('----------------Begin--------------------')
        # Loading checkpoint
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Restored checkpoint manager !')

        for epoch in range(self.epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            
            for (batch, (inp, tar)) in enumerate(data):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.3f} Accuracy {self.train_accuracy.result():.3f}')

                if (epoch + 1) % 5 == 0:
                    saved_path = self.checkpoint_manager.save()
                    print('Checkpoint was saved at {}'.format(saved_path))
        print('----------------Done--------------------')

    def predict(self, encoder_input, decoder_input, is_train, max_length, end_token):
        """
        Dự đoán câu đích dựa trên câu nguồn.

        Args:
            encoder_input (tensor): Câu nguồn (source sentence), shape: (batch_size, src_len)
            decoder_input (tensor): Câu đích (target sentence), shape: (batch_size, targ_len)
            is_train (bool): Có phải chế độ huấn luyện hay không
            max_length (int): Độ dài tối đa của câu đích
            end_token (int): Token kết thúc câu đích

        Returns:
            tensor: Câu đích dự đoán, shape: (batch_size, targ_len)
        """
        print('=============Inference Progress================')
        print('----------------Begin--------------------')
        # Loading checkpoint
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Restored checkpoint manager !')
        
        for i in range(max_length):

            encoder_padding_mask, decoder_look_ahead_mask ,decoder_padding_mask = generate_mask(encoder_input, decoder_input)

            preds = self.model(encoder_input, decoder_input, is_train, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask)
            # print('---> preds', preds)

            preds = preds[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(preds, axis=-1)

            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == end_token:
                break

        return decoder_input