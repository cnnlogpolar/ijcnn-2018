import glob
import numpy
import os
from scipy.misc import imread
from scipy.misc import imresize

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

########################### LOADING IMAGE DATA ################################
train_dir = 'TUDarmstadt_train/PNGImages/'
test_dir = 'TUDarmstadt_test_rot45/PNGImages/'
# Image settings
h = 224  # height
w = 224  # width
dp = 3  # color RGB
classes = ['car', 'cow', 'motorbike', ]

def readimage(directory):
    
    img_paths = glob.glob(directory + '**/*.png', recursive=True)
    n_imgs = len(img_paths) # number of images
    imgs = numpy.zeros((n_imgs,h,w,dp), dtype=numpy.float32)
    
    
    listfile =[]
    listot =[]
    labels=[]
    j=0
    for root, dirnames, filenames in os.walk(directory):
        for dirname in dirnames:
            addrs = glob.glob(root + dirname + '/*.png')
            for i, val in enumerate(addrs):
                #labels.append(len(listfile)+1)
                labels.append(dirname)
                img = imread(val)
        
                img = imresize(img,(h,w))    
                imgs[j, ...] = img
                j=j+1
            listfile.append(addrs)
            print('read files:' + root + dirname)
    
    #labels=LabelBinarizer().fit_transform(labels)        
    for item in listfile:
        listot = listot + item
        
    return imgs,labels

print('Loading training set')
X_train,y_train = readimage(train_dir)
print('Loading test set')
X_test,y_test = readimage(test_dir)

########################### ENCODING CLASSES ################################
# Import `StandardScaler` from `sklearn.preprocessing`
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

### Change any labels to sequential integer labels
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
Yn_train = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
Yi_train = np_utils.to_categorical(Yn_train)
uniques, ids = numpy.unique(Yn_train, return_inverse=True)
n_classes = len(uniques)
Yn_test = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
Yi_test = np_utils.to_categorical(Yn_test)

print(n_classes)
print(uniques)
########################### NORMALIZATION ####################################
X_train /= 255 # ou gaussian
X_test /= 255

############################################ CONVOLUTIONAL NEURAL NETWORK ######################################################

# Import libraries and modules
from keras.models import Sequential
from keras.layers import  Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from logpolar_simple import LogPolar
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras import backend as K
from sklearn.metrics import average_precision_score

class MetricsCallback(Callback):
    def __init__(self,  train_data, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.train_data = train_data
        self.ap_scores = []
        self.cutoff = .5

    def on_epoch_end(self, epoch, logs={}):
        X_val = self.validation_data[0]
        y_val = self.validation_data[1]

        preds = self.model.predict(X_val)

        #f1 = f1_score(y_val, (preds > self.cutoff).astype(int))
        ap= average_precision_score(y_val, (preds > self.cutoff).astype(int))
        print("\n MAP for epoch %d is %f"%(epoch, ap))
        self.ap_scores.append(ap)

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall



def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    return precision

# Load data into train and test sets

# Preprocess input data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Preprocess class labels
Y_train = Yi_train
Y_test = Yi_test

def build_cnn_alex(l1,l2):
    """"
    The first convolutional layer filters the 224×224×3 input image with 96 kernels of size 11×11×3
    with a stride of 4 pixels (this is the distance between the receptive field centers of neighboring
    neurons in a kernel map). The second convolutional layer takes as input the (response-normalized
    and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5 × 5 × 48.
    The third, fourth, and fifth convolutional layers are connected to one another without any intervening
    pooling or normalization layers. The third convolutional layer has 384 kernels of size 3 × 3 ×
    256 connected to the (normalized, pooled) outputs of the second convolutional layer. The fourth
    convolutional layer has 384 kernels of size 3 × 3 × 192 , and the fifth convolutional layer has 256
    kernels of size 3 × 3 × 192. The fully-connected layers have 4096 neurons each.

    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    """

    # Define model architecture
    model = Sequential()

    #AlexNet with batch normalization in Keras 
    #input image is 224x224

    if l1 == 'LPC':
        model.add(LogPolar(input_shape=(h, w, dp))) # 200x150x3 => 200x150x3
        model.add(Conv2D(64, (11, 11)))
    if l1 == 'LP':
        model.add(LogPolar(input_shape=(h, w, dp))) # 200x150x3 => 200x150x3
    if l1 == 'C':
        model.add(Conv2D(64, (11, 11), input_shape=(h, w, dp)))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    if l2 == 'LPC':
        model.add(LogPolar()) # 200x150x3 => 200x150x3
        model.add(Conv2D(128, (7, 7)))
    if l2 == 'LP':
        model.add(LogPolar()) # 200x150x3 => 200x150x3
    if l2 == 'C':
        model.add(Conv2D(128, (7, 7)))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(192, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4096, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(n_classes, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    # Compile model
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def build_cnn(l1,l2):
    model = Sequential()
    # logpolar layer
    if l1 == 'LPC':
        model.add(LogPolar(input_shape=(h, w, dp))) # 200x150x3 => 200x150x3
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) # 200x150x3 => 100x75
    if l1 == 'LP':
        model.add(LogPolar(input_shape=(h, w, dp))) # 200x150x3 => 200x150x3
        model.add(MaxPooling2D(pool_size=(2, 2))) # 200x150x3 => 100x75
    if l1 == 'C':
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, dp))) # 200x150x3 => 198x148x32
        model.add(MaxPooling2D(pool_size=(2, 2))) # 198x148x32 => 99x74x32
    if l2 == 'LPC':
        model.add(LogPolar()) # 200x150x3 => 200x150x3
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) # 200x150x3 => 100x75
    if l2 == 'LP':
        model.add(LogPolar()) # 99x74x32 => 99x74x32
        #model.add(Lambda(LogPolarLambda, output_shape=LogPolarLambda_output_shape))  # 99x74x32 => 99x74x32
        model.add(MaxPooling2D(pool_size=(2, 3)))  # 99x74x32 => 33x37x32
    if l2 == 'C':
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) # 200x150 => 100x75
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




## Model 1 : Conv2D + ReLU + MaxPooling + Conv2D + ReLU + MaxPooling + Dropout + Flatten + Dense + ReLU + Dropout + Dense + Softmax
model = build_cnn('C','LPC')

#tensorboard = TensorBoard(log_dir='logs/',  write_images=True,write_graph=True, write_grads=True, histogram_freq=1)

print('Training')
# Fit model on training data
metrics_callback = MetricsCallback(train_data=(X_train, Y_train), validation_data=(X_test, Y_test))

#model.fit(X_train, Y_train, batch_size=32, epochs=20, validation_data=(X_test, Y_test),verbose=1,callbacks=[metrics_callback,tensorboard])
model.fit(X_train, Y_train, batch_size=32, epochs=20, validation_data=(X_test, Y_test),verbose=1,callbacks=[metrics_callback])

print(model.metrics_names)

print('Testing')
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
print('MF - C-LPC -rot45')

