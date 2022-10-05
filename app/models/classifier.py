import pickle
import numpy as np
import cv2
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

from keras import backend
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Sequential
from keras import backend


dog_names = pickle.load(open("models/dog_names.p", "rb"))

face_cascade = cv2.CascadeClassifier(
    'models/haarcascades/haarcascade_frontalface_alt.xml')

resnet_model = load_model()

def load_model():
    """
    Load Resnet architecture and pretrained model
    """
    backend.clear_session()
    resnet_model = Sequential()
    resnet_model.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
    resnet_model.add(Dense(133, activation='softmax'))

    resnet_model.load_weights('models/saved_models/weights.best.resnet50.hdf5')


def face_detector(img_path):
    """
    Input:
        img_path: a string-valued file path to an image

    Output:
        Returns True if a human face is detected in an image and False otherwise
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):
    """
    Input:
        img_path: a string-valued file path to an image

    Output:
        Returns a 4D tensor with shape (1,224,224,3)
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def extract_Resnet50(tensor):
    """
    Input:
        tensor: a tensor of an image

    Output:
        Returns a feature for the object that is extracted from the image
    """
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def dog_detector(img_path):
    """
    Input:
        img_path: a string-valued file path to an image

    Output:
        Returns "True" if a dog is detected in the image
    """
    # download pre-trained Resnet model along with weights that have been trained on ImageNet
    ResNet50_model = ResNet50(weights='imagenet')

    # get prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(ResNet50_model.predict(img))

    # get the categories corresponding to dogs that correspond to dictionary keys 151-268
    return ((prediction <= 268) & (prediction >= 151))


def predict_breed(img_path):  
    """
    Input:
        img_path: a string-valued file path to an image

    Output:
        Returns the dog breed that is predicted by the resnet_model
    """ 
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    
    # obtain predicted vector
    pred_vector = resnet_model.predict(bottleneck_feature)
    
    # return predicted dog breed
    return dog_names[np.argmax(pred_vector)]

def dog_breed_detector(img_path):
    """
    Input:
        img_path: a string-valued file path to an image

    Output:
        - if a dog is detected in the image, return the predicted breed.
        - if a human is detected in the image, return the resembling dog breed.
        - if neither is detected in the image, provide output that indicates an error.
    """
    
    if dog_detector(img_path):
        prediction = predict_breed(img_path).partition('.')[-1]
        return f"A dog is detected in the image, and its breed is: {prediction}" 
    
    if face_detector(img_path):
        prediction = predict_breed(img_path).partition('.')[-1]
        return f"A human is detected in the image, but the resembling dog breed is: {prediction}"

    return "Couldn't recognize a human or dog in the image. Please try another one."


