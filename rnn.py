import numpy as np
import tensorflow as tf

def simple_var_model(train_x, train_y, test_x, test_y, n, seq_len, num_hidden=2):
    with tf.Graph().as_default():
        batch_size = len(train_x)
        # batch_size = 10
        data = tf.placeholder(tf.float32, [None, seq_len, n])
        target = tf.placeholder(tf.float32, [None, n])
        bias = tf.Variable(tf.constant(0.5, shape=[target.get_shape()[1]]))
        cell = tf.contrib.rnn.BasicRNNCell(num_hidden, activation=tf.nn.relu)
        # val, _ = tf.nn.dynamic_rnn(cell, data, sequence_length=[seq_len]*batch_size, initial_state=cell.zero_state(batch_size, tf.float32)+bias, dtype=tf.float32)
        val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0])-1)
        # weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
        # prediction = tf.matmul(last, weight)
        prediction = last
        sqerr = tf.nn.l2_loss(tf.subtract(target, prediction))
        optimizer = tf.train.AdamOptimizer(0.01)
        minimize = optimizer.minimize(sqerr)

        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        num_batches = len(train_x)/batch_size
        epochs = 5000
        train_error = []
        for i in range(epochs):
            for j in range(num_batches):
                train_input, train_output = train_x[j*batch_size:(j+1)*batch_size], train_y[j*batch_size:(j+1)*batch_size]
                sess.run(minimize, {data: train_input, target: train_output})
                train_error.append(sess.run(sqerr, {data: train_input, target: train_output}))
        # test_prediction = []
        # for j in range(len(test_x)/batch_size):
        #     test_prediction.append(sess.run(prediction, {data: test_x[j*batch_size:(j+1)*batch_size], target: test_y[j*batch_size:(j+1)*batch_size]}))
        test_prediction = sess.run(prediction, {data: test_x, target: test_y})
        test_err = np.sqrt(np.sum((np.array(test_prediction)-np.array(test_y))**2)/len(test_prediction))
        sess.close()
        return test_err, train_error, test_prediction
