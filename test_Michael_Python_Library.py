import unittest
import Michael_Python_Library as MPL
import pandas as pd
import random


def length_of_predict(num_data_entries, num_predictions):
    x = pd.DataFrame([random.random() for _ in range(num_data_entries)])
    y = pd.DataFrame([random.random() for _ in range(num_data_entries)])
    x_predict = [random.random() for _ in range(num_predictions)]

    polynomial_regression = MPL.PolynomialRegression(x, y, 3)

    return len(polynomial_regression.predict(x_predict))


class TestMichaelPythonLibrary(unittest.TestCase):

    def test_prediction_length(self):
        self.assertEqual(length_of_predict(5, 3), 3)
        self.assertEqual(length_of_predict(10, 3), 3)
        self.assertEqual(length_of_predict(2, 10), 10)
        self.assertEqual(length_of_predict(1290, 100), 100)
        self.assertEqual(length_of_predict(1, 99), 99)
        self.assertEqual(length_of_predict(2, 4), 4)


if __name__ == '__main__':
    unittest.main()