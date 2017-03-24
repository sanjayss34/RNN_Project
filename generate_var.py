import numpy as np

def simple_var(n, T, sequence_len=10):
    y = np.zeros((T, n))
    x = np.random.rand(T, n)
    alpha = np.random.rand()
    num_seq = T/sequence_len
    y_0 = np.random.rand(1, n)
    for s in range(num_seq):
        start = s*sequence_len
        y[start] = y_0
        for t in range(1, sequence_len):
            y[t+start] = alpha*y[start+t-1]+(1-alpha)*x[start+t]
    i = 0
    test_num = num_seq/10
    train_x = []
    train_y = []
    while i < T-(test_num*sequence_len):
        add = min(sequence_len, T-(test_num*sequence_len)-i)
        train_x.append(x[i:i+add])
        train_y.append(y[i+add-1])
        i += add
    test_x = [x[j:j+sequence_len] for j in range(i, T, sequence_len)]
    test_y = y[i+sequence_len-1::sequence_len]
    return train_x, train_y, test_x, test_y, y, alpha

def linear_var(n, T, sequence_len=10):
    y = np.zeros((T, n))
    x = np.random.rand(T, n)
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    num_seq = T/sequence_len
    y_0 = np.random.rand(1, n)
    for s in range(num_seq):
        start = s*sequence_len
        y[start] = y_0
        for t in range(1, sequence_len):
            y[t+start] = np.dot(y[start+t-1], A) + np.dot(x[start+t], B)
    i = 0
    test_num = num_seq/10
    train_x = []
    train_y = []
    while i < T-(test_num*sequence_len):
        add = min(sequence_len, T-(test_num*sequence_len)-i)
        train_x.append(x[i:i+add])
        train_y.append(y[i+add-1])
        i += add
    test_x = [x[j:j+sequence_len] for j in range(i, T, sequence_len)]
    test_y = y[i+sequence_len-1::sequence_len]
    return train_x, train_y, test_x, test_y, A, B
