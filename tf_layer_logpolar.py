import numpy as np
import tensorflow as tf
import cv2

arq = '000001_rot180.jpg'
#arq = 'martalogpolar\\cartesian.png'


def tf_layers_logpolar(src):
    # Create tf variable for image
    img_tf = tf.Variable(src)

    # get image shape
    ihwd = img_tf.shape
    x_max = tf.constant(ihwd[0].value)
    y_max = tf.constant(ihwd[1].value)
    channels = tf.constant(ihwd[2].value)
    radius = 200
    max_theta = 275

    # Create placeholders for other parameters of LogPolar
    #img_xc = tf.placeholder(tf.int32,shape=(1))
    #img_yc = tf.placeholder(tf.int32,shape=(1))
    img_xc = tf.subtract(tf.divide(x_max, 2), 1)
    img_yc = tf.subtract(tf.divide(y_max, 2), 1)

    img_xc = tf.cast(img_xc, tf.int32)
    img_yc = tf.cast(img_yc, tf.int32)

    # Create Cartesian to LogPolar transformation
    # calculate the maximum radius to iterate
    xr_max = tf.maximum(img_xc, tf.subtract(x_max, img_xc))
    yr_max = tf.maximum(img_yc, tf.subtract(y_max, img_yc))
    # convert maximum delta x and y to radius in logpolar
    r_max = tf.sqrt(tf.to_float(tf.add(tf.square(xr_max), tf.square(yr_max))))

    # if radius not informed, take maximum radius (Nrho)
    #if tf.equal(radius,None):
        #radius = tf.ceil(r_max)

    # if theta not informed, take maximum theta (Ntheta)
    #if tf.equal(max_theta,None):
        #max_theta = yr_max

    # calculate steps to iterate and convert cartesian to logpolar
    rho_step = tf.divide(tf.log(r_max), tf.to_float(radius))
    theta_step = tf.divide(tf.multiply(2.0, np.pi), tf.to_float(max_theta))

    img_logpolar = tf.Variable(tf.cast(tf.zeros((radius,max_theta,channels), tf.float32), tf.uint8), tf.float32)
    img_logpolar = tf.reshape(img_logpolar,[-1])

    irho = tf.constant(0)
    rholoop = lambda irho,img_logpolar: tf.less(irho,radius)

    #tensorboard --logdir c:\users\frederico\documents\TUDarmstadt\graph#
    # writer = tf.summary.FileWriter(logs_path, graph=sess.graph)

    def rhoblk(irho,img_logpolar):
        rho = tf.exp(tf.multiply(tf.to_float(irho), tf.to_float(rho_step)))
        itheta = tf.constant(0)
        thetaloop = lambda itheta,img_logpolar: tf.less(itheta, max_theta)

        def thetablk(itheta,img_logpolar):
            theta = tf.multiply(tf.to_float(itheta), tf.to_float(theta_step))

            i = tf.to_int32(tf.round(tf.add(tf.to_float(img_xc), tf.multiply(rho, tf.sin(theta)))))
            j = tf.to_int32(tf.round(tf.add(tf.to_float(img_yc), tf.multiply(rho, tf.cos(theta)))))

            def update_image(img_logpolar):
                indices1 = irho * max_theta * channels + itheta * channels
                indices2 = indices1+1
                indices3 = indices1+2
                value1 = img_tf[i, j, 0]
                value2 = img_tf[i, j, 1]
                value3 = img_tf[i, j, 2]
                value1 = tf.reshape(value1, [1])
                value2 = tf.reshape(value2, [1])
                value3 = tf.reshape(value3, [1])
                img_logpolar = tf.concat(values=[img_logpolar[:indices1], value1, img_logpolar[(indices1+1):]], axis=0)
                img_logpolar = tf.concat(values=[img_logpolar[:indices2], value2, img_logpolar[(indices2+1):]], axis=0)
                img_logpolar = tf.concat(values=[img_logpolar[:indices3], value3, img_logpolar[(indices3+1):]], axis=0)
                img_logpolar = tf.reshape(img_logpolar,[radius*max_theta*3])
                return img_logpolar

            img_logpolar = tf.cond(tf.logical_and(tf.logical_and(tf.greater_equal(i,0),tf.less(i,tf.subtract(x_max,1))),tf.logical_and(tf.greater_equal(j,0),tf.less(j,tf.subtract(y_max,1)))),lambda: update_image(img_logpolar),lambda: img_logpolar)

            itheta = tf.add(itheta,1)
            return itheta, img_logpolar

        itheta,img_logpolar=tf.while_loop(thetaloop, thetablk, [itheta,img_logpolar])
        irho = tf.add(irho,1)
        return irho,img_logpolar

    irho,img_logpolar=tf.while_loop(rholoop, rhoblk, [irho,img_logpolar])

    img_logpolar = tf.reshape(img_logpolar,[radius,max_theta,channels])
    return irho, img_logpolar

## Main ###
src = cv2.imread(arq, cv2.IMREAD_COLOR)
cv2.namedWindow("inverse log-polar", 1)

irho, img_logpolar = tf_layers_logpolar(src)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


with tf.Session() as sess:
    sess.run(init_op)
    # img = sess.run(img_tf)
    # Run tf graphs

    img_logpolar_value = sess.run(img_logpolar)
    print(img_logpolar_value)

    #sess.run(irho)

    cv2.namedWindow("inverse log-polar", 1)
    cv2.imshow("inverse log-polar", img_logpolar_value)
    cv2.imwrite("your_file.jpeg", img_logpolar_value)
    
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    sess.close()