import glob
import numpy
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

########################### LOADING IMAGE DATA ################################
# Image settings
h = 150 # height
w = 200 # width
dp = 3 # color RGB

# Load image data
from scipy.misc import imread
from scipy.misc import imresize
import os
import numpy as np

def readimage(directory):
    
    img_paths = glob.glob(directory + '**/*.jpg', recursive=True)
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


########################### ENCODING CLASSES ################################
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

base_dir = 'TUDarmstadt_marta/PNGImages/'
X,Y=readimage(base_dir)

#base_dir = 'TUDarmstadt_fred/PNGImages/'
base_dir = 'TUDarmstadt_marta/PNGImages/'
X1,Y1=readimage(base_dir)

### Change any labels to sequential integer labels
# encode class values as integers
encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

# integer encode
Y_encoded = encoder.fit_transform(Y)
Y_encoded = Y_encoded.reshape(len(Y_encoded), 1)
# binary encode
Y_onehot = onehot_encoder.fit_transform(Y_encoded)

# integer encode
Y1_encoded = encoder.fit_transform(Y1)
Y1_encoded = Y1_encoded.reshape(len(Y1_encoded), 1)
# binary encode
Y1_onehot = onehot_encoder.fit_transform(Y1_encoded)



########################### NORMALIZATION ####################################
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
from logpolar import LogPolar
from keras.callbacks import TensorBoard

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
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_cnn('C','C')


tensorboard = TensorBoard(log_dir='logs/',  write_images=True,write_graph=True, write_grads=True, histogram_freq=1)


#write_graph=True, write_grads=True, write_images=True
# Fit model on training data
history =model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1,validation_data=(X_test, Y_test),callbacks=[tensorboard])

#history =model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

print(model.metrics_names)
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

y_predv = model.predict(X_test)

