import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

def show_predict(pic_num):
    class MNISTLoader():
        def __init__(self):
            cifar10 = tf.keras.datasets.cifar10
            (self.train_data, self.train_label), (self.test_data, self.test_label) = cifar10.load_data()


            # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
            self.train_data = self.train_data.astype(np.float32) / 255.0      # [50000, 32, 32, 3]
            self.test_data = self.test_data.astype(np.float32) / 255.0        # [10000, 32, 32, 3]
            # self.test_label.shape(10000,1) dtype=unit8
            # self.test_label.shape(50000,1) dtype=unit8
            self.train_label = self.train_label.astype(np.int32)    # [60000,1]
            self.test_label = self.test_label.astype(np.int32)      # [10000,1]
            self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

        def get_batch(self, batch_size):
            # 从数据集中随机取出batch_size个元素并返回
            index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
            return self.train_data[index, :], self.train_label[index]


    class CNN(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv2D(
                filters=32,             # 卷积层神经元（卷积核）数目
                kernel_size=[5, 5],     # 感受野大小
                padding='same',         # padding策略（vaild 或 same）
                activation=tf.nn.relu   # 激活函数
            )
            self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
            self.conv2 = tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=[5, 5],
                padding='same',
                activation=tf.nn.relu
            )
            self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
            self.flatten = tf.keras.layers.Reshape(target_shape=(8* 8 * 64,))
            self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(units=10)

        def call(self, inputs):
            x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
            x = self.pool1(x)                       # [batch_size, 14, 14, 32]
            x = self.conv2(x)                       # [batch_size, 14, 14, 64]
            x = self.pool2(x)                       # [batch_size, 7, 7, 64]
            x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
            x = self.dense1(x)                      # [batch_size, 1024]
            x = self.dense2(x)                      # [batch_size, 10]
            output = tf.nn.softmax(x)
            return output

    num_epochs = 1
    batch_size = 50
    learning_rate = 0.001


    model = CNN()
    data_loader = MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    # num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    num_batches = int(data_loader.num_train_data // batch_size)

    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            break
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


    model.load_weights('my_weights.h5')


    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())


    label_dict=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    a=model.predict(data_loader.test_data[pic_num:pic_num+1])
    b=np.arange(10)
    a=a.flatten()
    plt.figure()
    plt.bar(b, a)
    plt.xticks(list(range(10)),label_dict)

    plt.show()



    cifar10 = tf.keras.datasets.cifar10
    (train_data, train_label), (test_data, test_label) = cifar10.load_data()
    test_label = test_label.astype(np.int32) 
    cv2.namedWindow(label_dict[int(test_label[pic_num])],0)
    cv2.imshow(label_dict[int(test_label[pic_num])],test_data[pic_num])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# show_predict(5)