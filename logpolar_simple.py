from keras.engine.topology import Layer
from keras.layers import InputSpec
from keras import activations
import utilmyproj as util
import numpy as np
import tensorflow as tf

class LogPolar(Layer):

    def __init__(self, **kwargs):
        super(LogPolar, self).__init__(**kwargs)
        #self.input_spec = [InputSpec(ndim=4)]
        #self.activation = activations.get('linear')
        self.trainable = False

    def build(self, input_shape):
        #self.input_spec = [InputSpec(shape=input_shape)]
        #self.non_trainable_weights = [np.ones((input_shape[1],input_shape[2],input_shape[3])), np.ones((input_shape[3]))]
        super(LogPolar, self).build(input_shape)

    def call(self, x, **kwargs):
        def pre_logpolar(window):
            n, m = window.shape
            #radius, xc, yc, imgkey = util.aplicasift(img)
            radius = m / 2
            xc = m / 2
            yc = n / 2
            pre = util.aplicalogpolar(window, radius, xc, yc)
            return pre

        def process_filters(img):
            filters = pre_logpolar(img)
            return filters

        # Define Actual gradient
        def _logpolarGrad(op, grad):
            return grad

        def process_samples(inp):
            # needs: number of filters, kernel size, stride, padding, activation
            num, n, m, d = inp.shape
            res = np.zeros((num,n,m,d))
            for i in range(num):
                for c in range(d):
                    img = inp[i,...,c]
                    lpimg = process_filters(img)
                    res[i, ..., c] = lpimg
            return res.astype('float32')  #np.float32(res)

        grad = _logpolarGrad
        name = 'process_logpolar'
        stateful = True
        num, n, m, d = x.get_shape().as_list()
        # Need to generate a unique name to avoid duplicates:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(_logpolarGrad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            outputs = tf.reshape(tf.py_func(process_samples, [x], tf.float32, stateful=stateful, name=name),
                                 (-1, n, m, d))

        #outputs.set_shape(inputs.get_shape())
        #num, n, m, d = x.get_shape().as_list()
        #outputs = self.activation(tf.reshape(outputs,[-1, n, m, d]))
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
