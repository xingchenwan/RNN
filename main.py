from utils import *
from rnn import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_length = 10000
    train_data_proportion = 0.99
    num_segment = 50

    input_size = 1
    num_steps = 50
    num_epoch = 10

    # Segment the train and test data
    train_x, train_y = [], []
    for i in range(num_epoch):
        x, y = synthetic_series(train_length, input_size, num_steps)
        # x, y = hochreiter_schmidhuber(num_steps, 10000)
        # x, y = xor_series(num_steps, 10000)
        train_x.append(x)
        train_y.append(y)

    # test_x, test_y = hochreiter_schmidhuber(num_steps, 10000)
    # test_x, test_y = xor_series(num_steps, 100)
    test_x, test_y = synthetic_series(200, input_size, num_steps)
    # Initialise a RNN model

    params = {'batch_size': 64,
              'input_dimension': input_size,
              'num_steps': num_steps,
              'num_cells': 100,
              'learning_rate': 0.001,
              'learning_rate_decay_coeff': 0.99,
              'num_layers': 1,
              'num_epoch': num_epoch,
              'model_type': 'rnn'
              }
    rnn = RNN(train_x, train_y, test_x, test_y, **params)
    rnn.create_graph()
    rnn.run()
    rnn.gen_summary()
    plt.show()