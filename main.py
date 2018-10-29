from utils import *
from rnn import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_length = 1000
    train_data_proportion = 0.75
    num_segment = 50

    input_size = 1
    num_steps = 3


    # Segment the train and test data
    # x, y = synthetic_series(train_length, input_size, num_steps)

    x, y = hochreiter_schmidhuber(3, 50000)

    train_x = x[:int(train_length*train_data_proportion), :]
    train_y = y[:int(train_length*train_data_proportion), :]
    test_x = x[int(train_length*train_data_proportion):, :]
    test_y = y[int(train_length*train_data_proportion):, :]
    # Initialise a RNN model

    params = {'batch_size': 32,
              'input_dimension': input_size,
              'num_steps': num_steps,
              'num_cells': 128,
              'learning_rate': 0.001,
              'num_layers': 1,
              'model_type': 'rnn'
              }
    rnn = RNN([train_x], [train_y], test_x, test_y, **params)
    rnn.create_graph()
    rnn.run()
    rnn.gen_summary()
    plt.show()