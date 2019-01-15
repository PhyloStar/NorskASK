from keras.layers.pooling import _GlobalPooling1D
from keras import backend as K


class GlobalAveragePooling1D(_GlobalPooling1D):
    """Global average pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, features)`
    
    This layer is copied from a more recent version of Keras in order
    to support masking in a GlobalAveragePooling1D layer.
    The Keras version on Abel does not.
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
