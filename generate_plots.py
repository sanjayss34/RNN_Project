from generate_var import simple_var, linear_var
from rnn import simple_var_model
import matplotlib.pyplot as plt
import numpy as np

dim = 10
seq_len = 10
T = 1000
train_x, train_y, test_x, test_y, _, _ = linear_var(dim, T, seq_len)
n_vals = range(2, 11)
errs = []
for n in n_vals:
    errs.append(simple_var_model(train_x, train_y, test_x, test_y, dim, seq_len, num_hidden=n))
    print(str(n) + ' done')
plt.plot(n_vals, errs, 'bo')
plt.xlabel('Number of hidden cells')
plt.ylabel('RMSE (Test)')
plt.title('RNN for Simple VAR: Test Error vs. Number of hidden cells')
plt.xlim(1, 11)
plt.ylim(0, 1)
plt.show()
