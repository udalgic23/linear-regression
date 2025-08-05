import numpy as np

def loss_function(data, b, m):
    x = data[:, 0]
    y = data[:, 1]
    return 1 / x.size * np.sum(np.square(y - (m * x + b)))

def gradient_descent(data, b_cur, m_cur, learning_rate):
    x = data[:, 0]
    y = data[:, 1]
    
    n = x.size
    y_pred = m_cur * x + b_cur
    error = y - y_pred
    
    b_grad = -2 / n * np.sum(error)
    m_grad = -2 / n * np.sum(x * error)
    
    b_cur -= learning_rate * b_grad
    m_cur -= learning_rate * m_grad

    return b_cur, m_cur

def regression(data, b_init, m_init, num_iterations, learning_rate):
    b = b_init
    m = m_init

    for i in range(num_iterations):
        b, m = gradient_descent(data, b, m, learning_rate)
    return b, m

def run(data, b_init=0, m_init=0, num_iterations=10000, learning_rate=0.0001):
    print(f"initial loss = {loss_function(data, b_init, m_init)}")
    b_final, m_final = regression(data, b_init, m_init, num_iterations, learning_rate)
    print(f"final loss = {loss_function(data, b_final, m_final)}")
    print(f"final b = {b_final}, final m = {m_final}")

data = np.loadtxt("train.csv", delimiter=",", dtype=str)
data = data[1:].astype(float)

test = np.loadtxt("test.csv", delimiter=",", dtype=str)
test = data[1:].astype(float)

run(data, b_init=-0.03323, m_init=0.99954, num_iterations=10000)
