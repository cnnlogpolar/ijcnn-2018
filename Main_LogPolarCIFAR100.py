import glob
import numpy
from scipy.misc import imread
from scipy.misc import imresize

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
#https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
########################### LOADING IMAGE DATA ################################
# Image settings
h = 32 # height
w = 32 # width
dp = 3 # color RGB

# Load image data
import os

n_classes=100

def readimage(directory):
    
    img_paths = glob.glob(directory + '/**/*.png', recursive=True)
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

base_dir = 'cifar_train/'
X,Y=readimage(base_dir)


base_dir = 'cifar_test_rot90/'
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
from keras.layers import  Dense, Dropout,  Flatten
from keras.layers import Conv2D, MaxPooling2D
from logpolar_simple import LogPolar
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
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) # 200x150x3 => 100x75
        model.add(Dropout(0.25))

    if l1 == 'LP':
        model.add(LogPolar(input_shape=(h, w, dp))) # 200x150x3 => 200x150x3
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) # 200x150x3 => 100x75
        model.add(Dropout(0.25))

    if l1 == 'C':
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, dp))) # 200x150x3 => 198x148x32
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) # 198x148x32 => 99x74x32
        model.add(Dropout(0.25))

    if l2 == 'LPC':
        model.add(LogPolar()) # 200x150x3 => 200x150x3
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) # 200x150x3 => 100x75
        model.add(Dropout(0.25))
    if l2 == 'LP':
        model.add(LogPolar()) # 99x74x32 => 99x74x32
        model.add(Conv2D(64, (3, 3), activation='relu'))
        #model.add(Lambda(LogPolarLambda, output_shape=LogPolarLambda_output_shape))  # 99x74x32 => 99x74x32
        model.add(MaxPooling2D(pool_size=(2, 3)))  # 99x74x32 => 33x37x32
        model.add(Dropout(0.25))
    if l2 == 'C':
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) # 200x150 => 100x75
        model.add(Dropout(0.25))
		



    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = build_cnn('C','C')

# Fit model on training data
metrics_callback = MetricsCallback(train_data=(X_train, y_train), validation_data=(X_test, y_test))

model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),verbose=1,callbacks=[metrics_callback])

print(model.metrics_names)

print('Testing')
# Evaluate model on test data
score = model.evaluate(X_test, y_test, verbose=0)
print(score)
print('MF - C-C-90 ')
