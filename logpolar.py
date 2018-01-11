from keras.engine.topology import Layer
import utilmyproj2 as util
import numpy as np
import tensorflow as tf

class LogPolar(Layer):
    global xc,yc
    def __init__(self, **kwargs):
        super(LogPolar, self).__init__(**kwargs)        
        self.trainable = False

    def build(self, input_shape):
        super(LogPolar, self).build(input_shape)

    def call(self, x, **kwargs):
        def pre_logpolar(window):            
            n, m = window.shape[:2]
            #radius = m / 2
            xc = n / 2
            yc = m / 2
            #radius, xc, yc = util.aplicasift(window)
            #radius, xc, yc = util.aplicaCAMshift(window)
            pre = util.logpolar_naive(window, xc, yc)
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
                #iimg = inp[i,:n,:m,:d]
                #print(iimg.shape)
                #radius, xc, yc = util.aplicasift(iimg)
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
