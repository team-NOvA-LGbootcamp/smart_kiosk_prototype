import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

class Predictor:
    def __init__(self, age_model_path, gender_model_path):

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
        self.IMG_SIZE = 224

    def predict_image(self, img_list):
        output = []
        for img in img_list:
            # BGR을 RGB로 변경
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = img.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)  

            self.interpreter.set_tensor(self.input_details[0]['index'], img.astype(self.input_dtype))
            self.interpreter.invoke()
            
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            output.append(np.round(np.squeeze(output_data)))
        return output
