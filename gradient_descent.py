from numpy import genfromtxt as read_data


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
    points = read_data("data.csv", delimiter=',')
    (m, b) = gradient_descent(points, 1000, 0.0003)
    print("Line generated is y = {}x + {}".format(m, b))
    print("Prediction for 32 is {}".format(predict(32, m, b)))

if __name__ == '__main__':
    main()

