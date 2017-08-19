import numpy as np
import matplotlib.pyplot as plt


def plot(slope, intercept, points):

    """Plotting the points"""
    x = list()
    y = list()
    for p in points:
        x.append(p[0])
        y.append(p[1])
    plt.scatter(x, y)

    """Plotting the line"""
    x_min, x_max, y_min, y_max = plt.axis()
    x_values = [x_min, x_max]
    y_values = [predict(x, slope, intercept) for x in x_values]
    plt.plot(x_values, y_values, 'r')

    plt.show()


def predict(x, m, b):
    return (m * x) + b


def gradient_descent(points, num_iterations, learning_rate):
    m = 0
    b = 0
    for i in range(num_iterations):
        m_delta = 0
        b_delta = 0
        for j in range(len(points)):
            x = points[j, 0]
            y = points[j, 1]

            y_predict = predict(x, m, b)
            error = y - y_predict

            m_delta += (2/len(points)) * error * x
            b_delta += (2/len(points)) * error
        m += m_delta * learning_rate
        b += b_delta * learning_rate
    return [m, b]


def main():
    points = np.genfromtxt("data.csv", delimiter=',')
    (m, b) = gradient_descent(points, 1000, 0.0003)
    plot(m, b, points)

if __name__ == '__main__':
    main()

