# Xingchen Wan 2018 | xingchen.wan@st-annes.ox.ac.uk

import matplotlib.pyplot as plt
import numpy as np

# Three pathological series suggested by Hochreiter and Schmidhuber (1997) that are usually difficult for RNN trainings
# See details in the paper above and Martens and Sutskever (2011)


def hochreiter_schmidhuber(segment_length, num_segment, mode='add'):
    x = np.array([np.random.rand(segment_length, ) for _ in range(num_segment)])
    y = np.array([[np.array(0)] for _ in range(num_segment)])
    i = 0
    for segment in x:
        mark_idx = np.random.randint(0, len(segment), size=2)
        while mark_idx[0] == mark_idx[1]:
            mark_idx[1] = np.random.randint(0, len(segment), size=1)
        if mode == 'add':
            y[i][0] = segment[mark_idx[0]] + segment[mark_idx[1]]
        elif mode == 'multiply':
            y[i][0] = segment[mark_idx[0]] * segment[mark_idx[1]]
        else:
            raise ValueError()
        i += 1
    x = np.reshape(x, [x.shape[0], x.shape[1], 1])
    return x, y


def xor_series(segment_length, num_segment):
    x = np.array([np.array(np.random.choice(2, segment_length, p=[0.5, 0.5])) for _ in range(num_segment)])
    y = np.array([[np.array(0)] for _ in range(num_segment)])
    i = 0
    for segment in x:
        mark_idx = np.random.randint(0, len(segment), size=2)
        while mark_idx[0] == mark_idx[1]:
            mark_idx[1] = np.random.randint(0, len(segment), size=1)
        y[i][0] = segment[mark_idx[0]] ^ segment[mark_idx[1]]
        i += 1
    x = np.reshape(x, [x.shape[0], x.shape[1], 1])
    return x, y

# Non-pathological Series


def synthetic_series(series_length, input_size, num_steps):
    """
    Generate a series of 1D synthetic auto-correlative series
    :param series_length: length of the total series
    :param echo_step: number of steps the output lags behind the input
    :param num_segment: length of each segment signal for training purposes
    :return: tuple of input and output
    """
    idx = np.array([i for i in range(series_length)])
    x = np.exp(np.sin(idx) + np.cos(idx))
    #plt.plot(x)
    #plt.show()
    # Process the data
    x = [np.array(x[i * input_size: (i+1) * input_size]) for i in range(len(x) // input_size)]

    x = np.array([x[i: i+num_steps] for i in range(len(x) - num_steps)])
    y = np.array([x[i+num_steps] for i in range(len(x) - num_steps)])

    return x, y


def stock_series():
    pass
