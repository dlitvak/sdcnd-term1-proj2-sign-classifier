import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from lenet5 import Lenet


class TrafficSignClassifier:

    def __init__(self, X_train, y_train, X_valid, y_valid):
        self.EPOCHS = 20
        self.BATCH_SIZE = 128
        self.learning_rate = 0.001

        self.X_train, self.y_train = self.quick_normalize_img_data(X_train), y_train
        self.X_valid, self.y_valid = self.quick_normalize_img_data(X_valid), y_valid
        self.n_classes = np.max(y_train) + (1 if np.min(y_train) == 0 else 0)

        self.x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(self.y, self.n_classes)

        self.nn = Lenet()
        self.logits = self.nn.network(self.x, self.n_classes)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=self.logits)
        loss_operation = tf.reduce_mean(self.cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.saver = tf.train.Saver()

    def quick_normalize_img_data(self, x):
        return np.ndarray.astype((x - 128.0) / 128.0, np.float32)

    def evaluate(self, X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, self.BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset + self.BATCH_SIZE], y_data[offset:offset + self.BATCH_SIZE]
            accuracy = sess.run(self.accuracy_operation, feed_dict={self.x: batch_x, self.y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def train(self, overwrite=False):
        """Train the classifier"""
        if not overwrite:
            return

        X_train, y_train = shuffle(self.X_train, self.y_train)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)

            print("Training...")
            print()
            for i in range(self.EPOCHS):
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, self.BATCH_SIZE):
                    end = offset + self.BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(self.training_operation, feed_dict={self.x: batch_x, self.y: batch_y})

                validation_accuracy = self.evaluate(self.X_valid, self.y_valid)
                print("EPOCH {} ...".format(i + 1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()

            self.saver.save(sess, './lenet')
            print("Model saved")

    def test(self, X_test, y_test):
        X_test = self.quick_normalize_img_data(X_test)
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('.'))
            return self.evaluate(X_test, y_test)

    def classify(self, sign_img):
        sign_img = self.quick_normalize_img_data(sign_img)
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('.'))
            img_arr = np.reshape(sign_img, [1, *sign_img.shape])
            return sess.run(tf.arg_max(self.logits, 1), feed_dict={self.x: img_arr})

    def top_softmax_probs(self, img, top_num):
        img = self.quick_normalize_img_data(img)
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('.'))
            img_arr = np.reshape(img, [1, *img.shape])
            return sess.run(tf.nn.top_k(tf.nn.softmax(self.logits), k=top_num), feed_dict={self.x: img_arr})

    def outputFeatureMap(self, image_input, activation_min=-1, activation_max=-1):
        """
            Here make sure to preprocess your image_input in a way your network expects
            with size, normalization, ect if needed

            # Note: x should be the same name as your network's tensorflow data placeholder variable
            # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
        """
        image_input = self.quick_normalize_img_data(image_input)
        image_input = np.reshape(image_input, [1, *image_input.shape])
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('.'))
            for l, tf_activation in enumerate(self.nn.layers):
                fig = plt.figure(l+1, figsize=(15, 15))
                fig.suptitle("Layer " + str(l+1))
                activation = tf_activation.eval(session=sess,feed_dict={self.x : image_input})
                featuremaps = activation.shape[3]

                for featuremap in range(featuremaps):
                    plt.subplot(7,8, featuremap+1) # sets the number of feature maps to show on each row and column
                    plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
                    if activation_min != -1 & activation_max != -1:
                        plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
                    elif activation_max != -1:
                        plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
                    elif activation_min !=-1:
                        plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
                    else:
                        plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
                    plt.axis("off")


