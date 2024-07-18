
import cv2
import glob
import numpy as np

IMAGE_SIZE = 224

class Predictor:
    def __init__(self, model_path, env = "WINDOW"):
        self.env = env

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
