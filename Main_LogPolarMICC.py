import glob
import numpy


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
#https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
########################### LOADING IMAGE DATA ################################
# Image settings
h = 150 # height
w = 200 # width
dp = 3 # color RGB

# Load image data
from scipy.misc import imread
from scipy.misc import imresize
import os
from keras import backend as K
from keras.callbacks import Callback



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



def readimage(directory):
    
    img_paths = glob.glob(directory + '/**/*.jpg', recursive=True)
    n_imgs = len(img_paths) # number of images
    imgs = numpy.zeros((n_imgs,h,w,dp), dtype=numpy.float32)
    
    
    listfile =[]
    listot =[]
    labels=[]
    j=0
    for root, dirnames, filenames in os.walk(directory):
        for dirname in dirnames:
            addrs = glob.glob(root + dirname + '/*.jpg')
            for i, val in enumerate(addrs):
                #labels.append(len(listfile)+1)
                try:
                    labels.append(dirname)
                    img = imread(val)
            
                    img = imresize(img,(h,w))    
                    imgs[j, ...] = img
                    j=j+1
                except Exception: 
                    pass
            listfile.append(addrs)
            print('read files:' + root + dirname)
    
    #labels=LabelBinarizer().fit_transform(labels)        
    for item in listfile:
        listot = listot + item
        
    return imgs,labels


base_dir = '101_ObjectCategories/'
X,Y=readimage(base_dir)



base_dir = 'MICC-Flickr101/'
X1,Y1=readimage(base_dir)

########################### ENCODING CLASSES ################################
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

### Change any labels to sequential integer labels
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Yn = encoder.transform(Y)

encoder.fit(Y1)
Y1n = encoder.transform(Y1)

# convert integers to dummy variables (i.e. one hot encoded)
Y_onehot = np_utils.to_categorical(Yn)
Y1_onehot = np_utils.to_categorical(Y1n)


########################### NORMALIZATION ####################################
#X = (255*(X - np.max(X))/-np.ptp(X)).astype(int)
#X1 = (255*(X1 - np.max(X))/-np.ptp(X1)).astype(int)
X /= 255 # ou gaussian
X1 /= 255 # ou gaussian

### Split training and test
#X_train, X_test, y_train, y_test = train_test_split(X, Yi, test_size=0.20, random_state=42)
X_train,y_train=X,Y_onehot
X_test,y_test=X1,Y1_onehot

############################################ CONVOLUTIONAL NEURAL NETWORK ######################################################

# Import libraries and modules
from keras.models import Sequential
from keras.layers import Dense, Dropout,  Flatten
from keras.layers import Conv2D, MaxPooling2D


# Load data into train and test sets

# Preprocess input data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Preprocess class labels
Y_train = y_train
Y_test = y_test

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
    n_classes=101
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = build_cnn('C','C')

# Fit model on training data
metrics_callback = MetricsCallback(train_data=(X_train, Y_train), validation_data=(X_test, Y_test))

history =model.fit(X_train, Y_train, batch_size=50, epochs=100, validation_data=(X_test, Y_test),verbose=1,callbacks=[metrics_callback])

print(model.metrics_names)
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

y_predv = model.predict(X_test)
