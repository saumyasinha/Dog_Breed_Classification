import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer

class CNN:
    '''
    CNN classifier
    '''

    def __init__(self, train_x,train_y, test_x,test_y,n_classes, epochs=1, batch_size=2):
        '''
        Initialize CNN classifier data
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_x=train_x
        self.train_y = train_y
        self.test_x=test_x
        self.test_y = test_y
        self.n_classes=n_classes


    def conv_net(self,x):

        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv3 = tf.layers.conv2d(conv2, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        # layer_shape = conv2.get_shape()
        # num_features = layer_shape[1:4].num_elements()
        # fc1 = tf.reshape(conv2, [-1, num_features])
        fc1 = tf.contrib.layers.flatten(conv3)
        # fc1 = tf.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout
        fc1 = tf.layers.dropout(fc1, rate=0.25)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, self.n_classes)
        # print(out.get_shape())

        return out


    def evaluate(self):
        '''
        test CNN classifier and get MSE
        :return: MSE, test_y, predicted_y
        '''
        x = tf.placeholder(tf.float32, [None, 225, 225, 3])
        y = tf.placeholder(tf.float32, [None, self.n_classes])

        prediction = self.conv_net(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

        num_batches=int(len(self.train_x) / self.batch_size)
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                epoch_loss = 0
                for _ in range(num_batches):
                    i=_*self.batch_size
                    epoch_x, epoch_y = self.train_x[i:i+self.batch_size],self.train_y[i:i+self.batch_size]
                    # print(epoch_x.shape,epoch_y.shape)
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', self.epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: self.test_x, y: self.test_y}))


if __name__ == '__main__':

    input_train = np.load('train.npy')
    ID=np.load('ID.npy')

    label_dict={}
    with open('labels.csv') as f:
        for line in f:
            cols = line.split(',')
            label_dict[cols[0]]=cols[1].strip('\n')

    input_y=[label_dict[id] for id in ID]
    enc = LabelBinarizer()
    input_y_encoded=enc.fit_transform(input_y)


    image_indices = [i for i in range(len(input_train))]
    # shuffle(image_indices)

    split_index = int(0.8 * len(image_indices))
    train_x = input_train[:split_index]
    train_y = input_y_encoded[:split_index]

    test_x = input_train[split_index:]
    test_y = input_y_encoded[split_index:]
    n_classes=input_y_encoded.shape[1]

    # print(input_y_encoded[:2])
    cnn = CNN(train_x, train_y, test_x, test_y,n_classes)
    cnn.evaluate()