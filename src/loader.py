"""
MIT License

Copyright (c) 2012-2018 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files 
(the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, 
publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be 
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
loader
~~~~~~

A library to load the target image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

"""
https://github.com/unexploredtest/neural-networks-and-deep-learning/tree/master
"""
import csv
import cv2
import numpy as np
import random

def load_data():
    """Return the data as a tuple containing the training data,
    the validation data, and the test data.


    print(load_data_wrapper()[0])
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 1,925 entries.  Each entry is, in turn, a
    numpy ndarray with 361 values, representing the 19 * 19 = 361
    pixels in a single image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 1,925 entries.  Those entries are just the digit
    values (0,1,...,8,9,A,B,...Z) for the corresponding images contained 
    in the first entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 350 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below."""
    training_data =     ([], [])
    validation_data =   ([], [])
    test_data =         ([], [])

    archive_loc = "../data/c_archive"
    csv_loc = archive_loc + "/english.csv"

    with open(csv_loc, 'r') as file:
        csv_reader = csv.reader(file)
        validation_list = random_list()
        idx = 0
        for row in csv_reader:
            if (row[0] != "image"):
                # Format image data
                image = cv2.imread(archive_loc + row[0], 0)
                image_rev = cv2.bitwise_not(image)
                final = image_rev / 255.0

                # Fill out training data tuple
                training_data[0].append(final)
                training_data[1].append(row[1])

                #  Fill out validation data tuple and test data tuple
                if (validation_list[idx] == 1):
                    validation_data[0].append(final)
                    validation_data[1].append(row[1])

                    test_data[0].append(final)
                    test_data[1].append(row[1])
                
                idx = 0 if idx == 54 else (idx + 1)

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 1,925
    2-tuples ``(x, y)``.  ``x`` is a 361-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 35-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 361-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (361, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (361, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (361, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 35-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0,1,...,8,9,A,B,...,Z with no O) into a corresponding desired output from 
    the neural network."""
    if ('A' <= j <= 'N'):
        idx = ord(j) - ord('A') + 10
    elif ('P' <= j <= 'Z'):
        idx = ord(j) - ord('A') + 9
    else:
        idx = ord(j) - ord('0')
    e = np.zeros((35, 1))
    e[idx] = 1.0
    return e

def random_list():
    """Return a list that has 10 out of 55 elements set as a 1 and
    the rest are zero. The positions of the 10 1's are randomized in
    the list."""
    list = [0] * 55
    ones = 0
    while (ones < 10):
        random_int = random.randint(0, 54)
        if (list[random_int] != 1):
            list[random_int] = 1
            ones += 1