from keras import backend as K
from keras.layers.pooling import _GlobalPooling1D


class GlobalAveragePooling1D(_GlobalPooling1D):
    """Global average pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, features)`

    This class is copied from a recent version of Keras in order
    to support masking in a GlobalAveragePooling1D layer.
    The Keras version on Abel, 2.2.2, does not support masking in this
    class.

    COPYRIGHT

    All contributions by François Chollet:
    Copyright (c) 2015 - 2018, François Chollet.
    All rights reserved.

    All contributions by Google:
    Copyright (c) 2015 - 2018, Google, Inc.
    All rights reserved.

    All contributions by Microsoft:
    Copyright (c) 2017 - 2018, Microsoft, Inc.
    All rights reserved.

    All other contributions:
    Copyright (c) 2015 - 2018, the respective contributors.
    All rights reserved.

    Each contributor holds copyright over their respective contributions.
    The project versioning (Git) records all such contribution source information.

    LICENSE

    The MIT License (MIT)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, **kwargs):
        super(GlobalAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            input_shape = K.int_shape(inputs)
            broadcast_shape = [-1, input_shape[1], 1]
            mask = K.reshape(mask, broadcast_shape)
            inputs *= mask
            return K.sum(inputs, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None
