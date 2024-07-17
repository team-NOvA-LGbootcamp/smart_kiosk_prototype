
import numpy as np
import cv2
import tensorflow as tf
from keras.applications import MobileNetV2
from keras import *
from keras.layers import *
IMAGE_SIZE = 224

# Loading backbone
mobile_v2 = MobileNetV2(input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3), include_top = False, weights = "imagenet")

# Freezing the backbone weights
mobile_v2.trainable = False

# Creating a Age Network
age_net = Sequential([
    InputLayer((IMAGE_SIZE, IMAGE_SIZE, 3), name="ImageInput"),
    mobile_v2,
    Dropout(0.4, name = "SlightDroput"),
    Flatten(name="FlattenEmbeddings"),
    Dense(1, activation="linear", name="AgeOutput")
], name="AgeNet")

# Compiling Model
age_net.compile(
    loss = 'mae',
    optimizer = 'adam',
    weighted_metrics=[]
)

# Loading backbone
mobile_v2_g = MobileNetV2(input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3), include_top = False, weights = "imagenet")

# Freezing the backbone weights
mobile_v2_g.trainable = False

# Creating a Age Network
gender_net = Sequential([
    InputLayer((IMAGE_SIZE, IMAGE_SIZE, 3), name="ImageInput"),
    mobile_v2_g,
    Dropout(0.4, name = "SlightDroput"),
    GlobalAveragePooling2D(name="GlobalAvgPooling"),
    Dense(1, activation="sigmoid", name="AgeOutput")
], name="GenderNet")

# Compiling Model
gender_net.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'],
    weighted_metrics=[]
)


class Predictor:
    def __init__(self, age_model_path, gender_model_path, env = "WINDOW"):
        self.env = env
        print()
        print("model is loading")
        print()
        if  env == "WINDOW":
            from tensorflow.keras.models import load_model
            age_net.load_weights(age_model_path)
            gender_net.load_weights(gender_model_path)
            self.age_model = age_net
            self.gender_model = gender_net

        elif env == "RASPBERRY":
            import tflite_runtime.interpreter as tflite
            self.age_model_path = age_model_path
            self.gender_model_path = gender_model_path
            
            self.age_interpreter = tflite.Interpreter(model_path=age_model_path)
            self.age_interpreter.allocate_tensors()
            self.age_input_details = self.age_interpreter.get_input_details()
            self.age_output_details = self.age_interpreter.get_output_details()
            
            self.gender_interpreter = tflite.Interpreter(model_path=gender_model_path)
            self.gender_interpreter.allocate_tensors()
            self.gender_input_details = self.gender_interpreter.get_input_details()
            self.gender_output_details = self.gender_interpreter.get_output_details()
            
            self.input_dtype = self.input_details[0]['dtype']
        print()
        print("model is loaded")
        print()
        self.model_is_done = True
        self.IMG_SIZE = 224

    def predict_image(self, img_list):
        age_predictions = []
        gender_predictions = []
        
        for img in img_list:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0  # Normalize the image
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = img.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)  # Reshape for model input

            if self.env == "WINDOW":
                # Age prediction
                age_pred = self.age_model.predict(img,verbose=None)
        
                age_predictions.append(np.round(np.squeeze(age_pred))) # Assuming the age model uses a softmax output

                # Gender prediction
                gender_pred = self.gender_model.predict(img,verbose=None)
                gender_predictions.append(np.round(np.squeeze(gender_pred)))  # Assuming the gender model uses a sigmoid output

            # elif self.env == "RASPBERRY":
            #     output = []
            #     self.interpreter.set_tensor(self.input_details[0]['index'], img.astype(self.input_dtype))
            #     self.interpreter.invoke()
                
            #     output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            #     output.append(np.round(np.squeeze(output_data)))
        
            
        return age_predictions, gender_predictions
