# model_operation.py

class ModelOperation:
    def __init__(self, face_detection_model, face_recognition_model):
        # Initialize with instances of FaceDetectionModel and FaceRecognitionModel
        self.face_detection_model = face_detection_model
        self.face_recognition_model = face_recognition_model
    
    def process_image(self, image):
        # Process the input image to detect faces
        detected_faces = self.face_detection_model.detect_faces(image)
        
        # For simplicity, assume only one face is processed
        if detected_faces:
            recognized_profile = self.face_recognition_model.recognize_face(detected_faces[0])
            return recognized_profile
        else:
            return None
