import tensorflow as tf
import numpy as np


class RNN:
    def __init__(self, training_data, training_label, test_data, test_label, **options):

        self.training_data = np.array(training_data, dtype=np.float64)
        self.training_label = np.array(training_label, dtype=np.float64)
        self.test_data = np.array(test_data, dtype=np.float64)
        self.test_label = np.array(test_label, dtype=np.float64)

        # Sanity checks
        if self.training_data.shape[0] != self.training_label.shape[0]:
            raise ValueError("The length of training_data tensor does not match the training_label tensor!")
        if self.test_label.shape[0] != self.test_data.shape[0]:
            raise ValueError("The length of test_data tensor does not match the test_label tensor!")

        self.options = self.unpack_options(**options)

        # Data dimension of a single sample
        self.input_dimensions = self.training_data.shape[1:]
        self.num_labels = len(np.unique(self.training_label))
        self.graph = None
        self.loss = None
        self.optimizer = None
        self.predict = None
        self.tf_labels = None
        self.tf_dataset = None

    def create_graph(self):
        """
        Set up a computation graph for TensorFlow
        :return: None
        """
        self.graph = tf.Graph()
        model_type = self.options['model_type']
        optimiser_selected = self.options['optimizer']

        with self.graph.as_default():
            self.tf_dataset = tf.placeholder(tf.float32, shape=(None, *self.input_dimensions))
            self.tf_labels = tf.placeholder(tf.float32, shape=(None, self.num_labels))
            if model_type == 'rnn':
                logits = self.rnn_model()
            elif model_type == 'lstm':
                logits = self.lstm_model()
            else:
                raise NotImplementedError()

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=self.tf_labels))
            if optimiser_selected == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.options['learning_rate']).minimize(self.loss)
            elif optimiser_selected == 'grad':
                self.optimizer = tf.train.GradientDescentOptimizer(self.options['learning_rate']).minimize(self.loss)
            elif optimiser_selected == 'ada':
                self.optimizer = tf.train.AdagradOptimizer(self.options['learning_rate']).minimize(self.loss)

            self.predict = tf.nn.softmax(logits)

    def run(self):
        """
        Create a session according to the computation graph and run the model
        :return: None
        """
        if self.graph is None:
            raise ValueError("Create TensorFlow graph before running a session.")
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()

        # Stochastic gradient descent: train the data with a mini-batch each iteration
        batch_size = self.options['batch_size']
        batch_count = self.training_data.shape[0] % batch_size + 1
        for batch in range(batch_count):
            try:
                batch_data = self.training_data[batch*batch_size:(batch+1)*batch_size, :, :]
                batch_labels = self.training_label[batch*batch_size:(batch+1)*batch_size, :]
            except KeyError:
                batch_data = self.training_data[batch*batch_size:, :, :]
                batch_labels = self.training_label[batch*batch_size:(batch+1)*batch_size, :]

            feed_dict = {self.tf_dataset: batch_data, self.tf_labels: batch_labels}
            _, l, train_predictions = session.run([self.optimizer, self.loss, self.predict],
                                                  feed_dict=feed_dict)
            print('Batch number ', str(batch), '. Train accuracy: ',
                  str(self.get_accuracy(train_predictions, batch_labels)))
        test_feed_dict = {self.tf_dataset: self.test_data, self.tf_labels: self.test_label}
        _, test_predictions = session.run([self.loss, self.predict], feed_dict=test_feed_dict)
        print('Test accuracy: ', str(self.get_accuracy(test_predictions, self.test_data)))

    # Implementation of RNN and LSTM models
    def rnn_model(self):
        unstacked_data = tf.unstack(self.training_data, axis=1)
        num_layer = self.options['num_layer']
        num_cells = self.options['num_cells']
        if num_layer == 1:
            all_cells = tf.nn.rnn_cell.BasicRNNCell(num_cells)
        else:
            cells = []
            for i in range(num_layer):
                cell = tf.nn.rnn_cell.BasicRNNCell(num_cells,)
                cells.append(cell)
            all_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        outputs, state = tf.nn.static_rnn(all_cells, unstacked_data, dtype=tf.float32)
        output = outputs[-1]

        W = tf.Variable(tf.truncated_normal([num_cells, self.num_labels]))
        b = tf.Variable(tf.random_normal([self.num_labels]))
        logit = tf.matmul(output, W) + b
        return logit

    def lstm_model(self,):
        unstacked_data = tf.unstack(self.training_data, axis=1)
        num_layer = self.options['num_layer']
        num_cells = self.options['num_cells']

        if num_layer == 1:
            all_cells = tf.nn.rnn_cell.BasicLSTMCell(num_cells)
        else:
            cells = []
            for i in range(num_layer):
                cell = tf.nn.rnn_cell.BasicLSTMCell(num_cells, )
                cells.append(cell)
            all_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        outputs, state = tf.nn.static_rnn(all_cells, unstacked_data, dtype=tf.float32)
        output = outputs[-1]

        W = tf.Variable(tf.truncated_normal([num_cells, self.num_labels]))
        b = tf.Variable(tf.random_normal([self.num_labels]))
        logit = tf.matmul(output, W) + b
        return logit

    # Utility Functions
    @staticmethod
    def get_accuracy(y_star, y):
        correct_cnt = np.sum(np.argmax(y_star, 1) == np.argmax(y, 1))
        return 100. * correct_cnt / y_star.shape[0]

    @staticmethod
    def unpack_options(num_cells=24,
                       learning_rate=1e-6,
                       batch_size=100,
                       optimizer='adam',
                       model_type='rnn',
                       num_layer=1):

        options = {
            'num_cells': num_cells,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'model_type': model_type,
            'num_layer': num_layer
        }
        return options
