import tensorflow as tf


class Lenet:
    def __init__(self):
        self.layers = []

    def network(self, x, n_classes, grayscale=False):
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.1
        num_color_channles = 1 if grayscale else 3

        # Layer 1: Convolutional. Input = 32x32x3. Output = 30x30x6.
        F_W1 = tf.Variable(tf.truncated_normal([3,3,num_color_channles,6], mean = mu, stddev = sigma))
        F_b1 = tf.Variable(tf.zeros([6]))
        conv1 = tf.nn.bias_add(tf.nn.conv2d(x, F_W1, strides=[1,1,1,1], padding='VALID'), F_b1)

        # Activation.
        conv1 = tf.nn.relu(conv1)

        # Layer 2: Convolutional. Input = 30x30x6. Output = 26x26x16.
        F_W2 = tf.Variable(tf.truncated_normal([5,5,6,16], mean = mu, stddev = sigma))
        F_b2 = tf.Variable(tf.zeros([16]))
        conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, F_W2, strides=[1,1,1,1], padding='VALID'), F_b2)

        # Activation.
        conv2 = tf.nn.relu(conv2)

        # Pooling. Input = 26x26x16. Output = 13x13x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # Layer 3: Convolutional. Input = 13x13x16. Output = 12x12x26.
        F_W3 = tf.Variable(tf.truncated_normal([2,2,16,26], mean=mu, stddev=sigma))
        F_b3 = tf.Variable(tf.zeros([26]))
        conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, F_W3, strides=[1,1,1,1], padding='VALID'), F_b3)

        # Activation.
        conv3 = tf.nn.relu(conv3)

        # Layer 4: Convolutional. Input = 12x12x26. Output = 10x10x52.
        F_W4 = tf.Variable(tf.truncated_normal([3,3,26,52], mean=mu, stddev=sigma))
        F_b4 = tf.Variable(tf.zeros([52]))
        conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, F_W4, strides=[1,1,1,1], padding='VALID'), F_b4)

        # Activation.
        conv4 = tf.nn.relu(conv4)

        # Pooling. Input = 10x10x16. Output = 5x5x52.
        conv4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # Flatten. Input = 5x5x52. Output = 1300.
        conv4_flat = tf.reshape(conv4,  [-1, 1300])

        # Layer 5: Fully Connected. Input = 1300. Output = 400.
        w5 = tf.Variable(tf.truncated_normal([1300, 400], mean = mu, stddev = sigma))
        b5 = tf.Variable(tf.zeros([400]))
        fc1 = tf.add(tf.matmul(conv4_flat, w5), b5)

        # Activation.
        fc1 = tf.nn.relu(fc1)
        #fc1 = tf.nn.dropout(fc1, 0.50)

        # Layer 6: Fully Connected. Input = 400. Output = 120.
        w6 = tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma))
        b6 = tf.Variable(tf.zeros([120]))
        fc2 = tf.add(tf.matmul(fc1, w6), b6)

        # Layer 7: Fully Connected. Input = 120. Output = 84.
        w7 = tf.Variable(tf.truncated_normal([120, 84], mean = mu, stddev = sigma))
        b7 = tf.Variable(tf.zeros([84]))
        fc3 = tf.add(tf.matmul(fc2, w7), b7)

        # Activation.
        fc3 = tf.nn.relu(fc3)
        # fc2 = tf.nn.dropout(fc2, 0.50)

        # Layer 5: Fully Connected. Input = 84. Output = 10.
        W_out = tf.Variable(tf.truncated_normal([84, n_classes], mean = mu, stddev = sigma))
        b_out = tf.Variable(tf.zeros([n_classes]))
        logits = tf.add(tf.matmul(fc3, W_out), b_out)

        # expose layers
        self.layers = [conv1, conv2, conv3, conv4]

        return logits


    def lenet_lab(x, n_classes, grayscale=False):
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.1
        num_color_channles = 1 if grayscale else 3

        # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
        F_W1 = tf.Variable(tf.truncated_normal([5,5,num_color_channles,6], mean = mu, stddev = sigma))
        F_b1 = tf.Variable(tf.zeros([6]))
        conv1 = tf.nn.bias_add(tf.nn.conv2d(x, F_W1, strides=[1,1,1,1], padding='VALID'), F_b1)

        # Activation.
        conv1 = tf.nn.relu(conv1)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # Layer 2: Convolutional. Output = 10x10x16.
        F_W2 = tf.Variable(tf.truncated_normal([5,5,6,16], mean=mu, stddev=sigma))
        F_b2 = tf.Variable(tf.zeros([16]))
        conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, F_W2, strides=[1,1,1,1], padding='VALID'), F_b2)

        # Activation.
        conv2 = tf.nn.relu(conv2)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # Flatten. Input = 5x5x16. Output = 400.
        conv2_flat = tf.reshape(conv2,  [-1, 400])

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        w3 = tf.Variable(tf.truncated_normal([400,120], mean = mu, stddev = sigma))
        b3 = tf.Variable(tf.zeros([120]))
        fc1 = tf.add(tf.matmul(conv2_flat, w3), b3)

        # Activation.
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, 0.50)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        w4 = tf.Variable(tf.truncated_normal([120, 84], mean = mu, stddev = sigma))
        b4 = tf.Variable(tf.zeros([84]))
        fc2 = tf.add(tf.matmul(fc1, w4), b4)

        # Activation.
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, 0.50)

        # Layer 5: Fully Connected. Input = 84. Output = 10.
        W_out = tf.Variable(tf.truncated_normal([84, n_classes], mean = mu, stddev = sigma))
        b_out = tf.Variable(tf.zeros([n_classes]))
        logits = tf.add(tf.matmul(fc2, W_out), b_out)

        return logits
