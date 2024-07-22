
import cv2
import glob
import numpy as np
import time
from itertools import combinations

IMAGE_SIZE = 224

class Predictor:
    def __init__(self, model_path, env = "WINDOW"):
        self.env = env
        self.face_img_list = []
        self.age_predictions, self.gender_predictions = [],[]
        if  env == "WINDOW":
            from tensorflow.keras.models import load_model # type: ignore
            model_files = glob.glob(model_path+'*')
            self.model = [load_model(file) for file in model_files]

        elif env == "RASPBERRY":
            import tflite_runtime.interpreter as tflite # type: ignore
            # self.age_model_path = age_model_path
            # self.gender_model_path = gender_model_path
            
            # self.age_interpreter = tflite.Interpreter(model_path=age_model_path)
            # self.age_interpreter.allocate_tensors()
            # self.age_input_details = self.age_interpreter.get_input_details()
            # self.age_output_details = self.age_interpreter.get_output_details()
            
            # self.gender_interpreter = tflite.Interpreter(model_path=gender_model_path)
            # self.gender_interpreter.allocate_tensors()
            # self.gender_input_details = self.gender_interpreter.get_input_details()
            # self.gender_output_details = self.gender_interpreter.get_output_details()
            
            # self.input_dtype = self.input_details[0]['dtype']
        print("\033[91mmodel is loaded\033[0m")

        self.IMG_SIZE = 224

    def set_face_image_list(self,img_list):
        self.face_img_list = img_list
        
    def run(self):
        try:
            while True:
                self.age_predictions, self.gender_predictions = self.predict_image(self.face_img_list)
                time.sleep(0.2)
        except Exception as e:
            print(e)

    def get_prediction_result(self):
        return self.age_predictions, self.gender_predictions

    def predict_image(self, img_list):
        age_predictions = []
        gender_predictions = []
        
        for img in img_list:
            # Convert BGR to RGB
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0  # Normalize the image
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = img.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)

            if self.env == "WINDOW":
                prediction_results = np.array([model.predict(img, verbose=0) for model in self.model])
                result = np.round(np.squeeze(np.mean(prediction_results,axis=0)))

                age_predictions.append(round(result[0]))
                gender_predictions.append(round(result[1]))
                

            # elif self.env == "RASPBERRY":
            #     output = []
            #     self.interpreter.set_tensor(self.input_details[0]['index'], img.astype(self.input_dtype))
            #     self.interpreter.invoke()
                
            #     output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            #     output.append(np.round(np.squeeze(output_data)))
        
            
        return age_predictions, gender_predictions


class RelationPredictor:
    def __init__(self, model_path, env="WINDOWS"):
        self.env = env

        if env == "WINDOW":
            from tensorflow.keras.models import load_model  # type: ignore
            self.model = load_model(model_path)

            if not self.model:
                raise ValueError("No models were loaded. Check the model path and file extensions.")

        elif env == "RASPBERRY":
            import tflite_runtime.interpreter as tflite  # type: ignore
            # Placeholder for Raspberry Pi model loading
            self.model = None  # Replace with actual model loading logic

        print("\033[91mmodel is loaded\033[0m")

        self.IMG_SIZE = 224

    def preprocess_image(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        # img = img.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)
        return np.array([img])

    def map_age(self,age):
        if age <= 4: return 0
        elif age <= 19: return 1
        elif age <= 44: return 2
        elif age <= 59: return 3
        else: return 4

    def predict_image(self, face_images,ages,genders):
        if len(face_images.keys()) < 2:
            raise ValueError("Only One person in frame")
        
        face_id_combinations = list(combinations(face_images.keys(), 2))
        # print(face_id_combinations)
        relation_proportion = np.array([0.0, 0.0, 0.0])
        try:
            for face_id1, face_id2 in face_id_combinations:
                face_image_1 = face_images[face_id1]
                face_image_2 = face_images[face_id2]
                face_image_1 = self.preprocess_image(face_image_1)
                face_image_2 = self.preprocess_image(face_image_2)

                age1, age2 = ages[face_id1], ages[face_id2]
                age1_mapped = self.map_age(age1)
                age2_mapped = self.map_age(age2)

                gender1, gender2 = genders[face_id1], genders[face_id2]
 
                metadata = np.array([[age1_mapped, gender1, age2_mapped, gender2]], dtype='float32')

                input_data = [face_image_1, face_image_2, metadata]

                if self.env == "WINDOW":
                    model_output = self.model.predict(input_data, verbose=0)
                    
                    model_output = np.squeeze(model_output)
                    relation_proportion += model_output

                elif self.env == "RASPBERRY":
                    # Placeholder for Raspberry Pi model predictions
                    pass

            relation_proportion /= len(face_id_combinations)

            relation_prediction_result = {
                    "friend": relation_proportion[0],
                    "family": relation_proportion[1],
                    "couple": relation_proportion[2],
            }

        except Exception as e:
            print(e)
        
        return relation_prediction_result
