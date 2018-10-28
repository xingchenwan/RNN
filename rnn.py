# Xingchen Wan 2018 | xingchen.wan@st-annes.ox.ac.uk

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class RNN:
    """
    Recurrent Neural Network
    """
    def __init__(self, training_data, training_label,
                 test_data, test_label,
                 **options):
        """
        :param training_data: :param training_label: input training data set and label tensors
        :param test_data: :param test_label: test data set and ground truth label tensors
        :param options: (hyper)parameters of the neural network model. See method unpack_options for details on the
        full list of configurable options
        """

        self.training_data = np.array(training_data, dtype=np.float32)
        self.training_label = np.array(training_label, dtype=np.float32)
        self.test_data = np.array(test_data, dtype=np.float32)
        self.test_label = np.array(test_label, dtype=np.float32)

        # Sanity checks
        if self.training_data.shape[0] != self.training_label.shape[0]:
            raise ValueError("The length of training_data tensor does not match the training_label tensor!")
        if self.test_label.shape[0] != self.test_data.shape[0]:
            raise ValueError("The length of test_data tensor does not match the test_label tensor!")

        self.options = self.unpack_options(**options)

        if self.options['input_dimension'] is None:
            # Data dimension of a single sample
            self.input_dimensions = 1
        else:
            self.input_dimensions = self.options['input_dimension']

        self.graph = None
        self.loss = None
        self.optimizer = None
        self.predict = None
        self.tf_labels = None
        self.tf_dataset = None
        self.losses = []

    def create_graph(self):
        """
        Set up a computation graph for TensorFlow
        :return: None
        """
        self.graph = tf.Graph()
        model_type = self.options['model_type']
        optimiser_selected = self.options['optimizer']

        with self.graph.as_default():
            self.tf_dataset = tf.placeholder(tf.float32,
                                             shape=(None, self.options['num_steps'], self.input_dimensions))
            self.tf_labels = tf.placeholder(tf.float32, shape=(None, self.input_dimensions))

            # Forward pass
            if model_type == 'rnn':
                predictions = self.rnn_model(self.tf_dataset)
            elif model_type == 'lstm':
                predictions = self.lstm_model(self.tf_dataset)
            else:
                raise NotImplementedError("Unimplemented RNN model keyword")

            self.loss = tf.reduce_mean(tf.square(predictions - self.tf_labels))

            if self.options['regularisation_coeff'] > 0.:
                # Add in L2 penalty for regularisation if required
                penalty = self.options['regularisation_coeff'] * sum(tf.nn.l2_loss(var)
                                                                     for var in tf.trainable_variables())
                self.loss += penalty

            if self.options['use_customised_optimizer'] is False:
                if optimiser_selected == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(self.options['learning_rate'])
                elif optimiser_selected == 'grad':
                    self.optimizer = tf.train.GradientDescentOptimizer(self.options['learning_rate'])
                elif optimiser_selected == 'ada':
                    self.optimizer = tf.train.AdagradOptimizer(self.options['learning_rate'])
                elif optimiser_selected == 'rms':
                    self.optimizer = tf.train.RMSPropOptimizer(self.options['learning_rate'])
                else:
                    raise NotImplementedError("Unimplemented built-in optimiser keyword.")
            else:
                self.optimizer = self.options['customized_optimizer']

            self.minimise = self.optimizer.minimize(self.loss)

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
            batch_count = self.training_data.shape[0] // batch_size
            for batch in range(batch_count):
                try:
                    batch_data = self.training_data[batch*batch_size:(batch+1)*batch_size, :, :]
                    batch_labels = self.training_label[batch*batch_size:(batch+1)*batch_size, :]
                except KeyError:
                    batch_data = self.training_data[batch*batch_size:, :, :]
                    batch_labels = self.training_label[batch*batch_size:, :]

                feed_dict = {
                    self.tf_dataset: batch_data,
                    self.tf_labels: batch_labels}

                l, _, = session.run([self.loss, self.minimise], feed_dict=feed_dict)
                self.losses.append(l)

    # Implementation of RNN and LSTM models
    def rnn_model(self, training_data):
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

        outputs, state = tf.nn.dynamic_rnn(all_cells, training_data, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        output = outputs[-1]

        W = tf.Variable(tf.truncated_normal([num_cells, self.input_dimensions]))
        b = tf.Variable(tf.random_normal([self.input_dimensions]))
        logit = tf.matmul(output, W) + b
        return logit

    def lstm_model(self, training_data):
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
        outputs, state = tf.nn.dynamic_rnn(all_cells, training_data, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        output = outputs[-1]

        W = tf.Variable(tf.truncated_normal([num_cells, self.input_dimensions]))
        b = tf.Variable(tf.random_normal([self.input_dimensions]))
        logit = tf.matmul(output, W) + b
        return logit

    # Utility Functions
    @staticmethod
    def unpack_options(num_cells=24,
                       learning_rate=1e-3,
                       batch_size=100,
                       optimizer='rms',
                       model_type='rnn',
                       use_customized_optimizer=False,
                       customized_optimizer=None,
                       num_layers=1,
                       regularisation_coeff=0,
                       input_dimension=None,
                       num_steps=30):
        """
        :param num_cells: Number of hidden units per layer in the RNN/LSTM network
        :param learning_rate: initial learning rate
        :param batch_size: batch size
        :param optimizer: choice of the chosen optimiser ('rms', 'adam', etc)
        :param model_type: 'rnn' or 'lstm'
        :param use_customized_optimizer: bool - if True the optimizer object in customized_optimizer will be used instead
        :param customized_optimizer: optimizer object - if use_customized_optimizer is True, this optimizer will be used
        :param num_layers: number of layers in the RNN/LSTM
        :param regularisation_coeff: regularisation coefficient (a.k.a lambda)
        :param input_dimension: input dimension of the each data point. For scalar time series this value is 1
        :param num_steps: number of data points of each input sequence
        :return:
        """

        options = {
            'num_cells': num_cells,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'model_type': model_type,
            'num_layer': num_layers,
            'use_customised_optimizer': use_customized_optimizer,
            'customized_optimizer': customized_optimizer,
            'regularisation_coeff': regularisation_coeff,
            "input_dimension": input_dimension,
            'num_steps': num_steps
        }
        return options

    # Plotter Functions
    def plot_loss(self):
        if len(self.losses) == 0:
            raise ValueError("The model session has not been run!")
        plt.plot(self.losses)
        plt.ylabel("Loss")
        plt.xlabel("Number of batch iterations")

