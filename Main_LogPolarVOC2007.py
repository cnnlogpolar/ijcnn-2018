import glob
import numpy

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
from xml.dom.minidom import parse




import os
listlabel=['person','bird','cat','cow','dog','horse','sheep','aeroplane','bicycle','boat','bus','car','motorbike','train','bottle','chair','dining','potted','sofa','tv']



def readimage(directory_img,directory_lbl):
   
    
    lbl_paths = glob.glob(directory_lbl + '**/*.xml' , recursive=True)

    n_imgs = len(lbl_paths) # number of images
    imgs = numpy.zeros((n_imgs,h,w,dp), dtype=numpy.float32)
    lbls = ['' for i in range(n_imgs)]

    # Load image data
    for i in range(0,n_imgs):
        if i % 100 == 0:
            print('Loading image #%05d\n' % (i+1))
        # Read metadata and label associated with images
        lbl_path = lbl_paths[i]        
        
        doc = parse(lbl_path)
        collection = doc.documentElement
        anots = collection.getElementsByTagName("name")
        
        for tag in anots:
            if (tag.parentNode.nodeName=='object'):
                img_label =tag.firstChild.nodeValue
                break
             
        #img_label =  anots.getElementsByTagName("name")[0].firstChild.nodeValue
        
        bachou=0
        for val in listlabel:
            if (img_label.find(val)!=-1):
                img_label=val
                bachou=1
                break
            
        if (bachou==0) :
            print("WARNING!!!  -> The file's label " + lbl_path + " was not foundS.")
        
        lbls[i] = img_label
    
        # Read image files from directory
        
        img_path = os.path.basename(directory_img + doc.getElementsByTagName('filename')[0].firstChild.nodeValue)
        
        
        img = imread(directory_img + img_path)
        
        img = imresize(img,(h,w))
        
        imgs[i, ...] = img
        
    
    # Define X and Y from image data
    X = imgs
    Y = lbls

        
    return X,Y


########################### ENCODING CLASSES ################################
from time import gmtime, strftime

base_dir = 'VOC2007_train/JPEGImages/'
lbldir='VOC2007_train/Annotations/'
print("lendo base de treino - " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
X,Y=readimage(base_dir,lbldir)


base_dir = 'VOC2007_test_rot90/JPEGImages/'
lbldir='VOC2007_test_rot90/Annotations/'
print("lendo base de teste" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
X1,Y1=readimage(base_dir,lbldir)

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

### Change any labels to sequential integer labels
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Yn = encoder.transform(Y)
uniques, ids = numpy.unique(Yn, return_inverse=True)
n_classes = len(uniques)

encoder.fit(Y1)
Y1n = encoder.transform(Y1)

# convert integers to dummy variables (i.e. one hot encoded)
Y_onehot = np_utils.to_categorical(Yn)
Y1_onehot = np_utils.to_categorical(Y1n)

lista=Y1n.reshape(1, -1).tolist()


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
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
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


model = build_cnn('C','LPC')

print("iniciando treinamento..." + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# Fit model on training data
metrics_callback = MetricsCallback(train_data=(X_train, Y_train), validation_data=(X_test, Y_test))

history =model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test),verbose=1,callbacks=[metrics_callback])

print(model.metrics_names)

print("iniciando teste..." + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)

print(score)

print('C-LPC')
