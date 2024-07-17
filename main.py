import cv2
import threading
from face_detection import FaceDetection
from model_operation import Predictor
from recommendation import RecommendationAlgorithm
import tensorflow as tf
import os


def main():

    face_detection = FaceDetection(camera_index=0)
    face_detection_run_thread = threading.Thread(target=face_detection.run)
    face_detection_run_thread.start()

    age_model_path = "./smart_kiosk_prototype/age_mobv2.h5"
    gender_model_path = "./smart_kiosk_prototype/Gender_mobv2.h5"

    if os.path.isfile(age_model_path) and os.path.isfile(gender_model_path):
        model_operation = Predictor(age_model_path,gender_model_path,env="WINDOW")

    recommendation = RecommendationAlgorithm()

    try:
        while True:
            
            org_frame = face_detection.get_org_frame()
            face_frames = face_detection.get_face_frames()
            face_bboxes = face_detection.get_face_bboxes()
            
            if org_frame is not None:
                # Convert the frame back to BGR for OpenCV display
                display_frame = cv2.cvtColor(org_frame, cv2.COLOR_RGB2BGR)
                
                age_prediect, gender_predict = model_operation.predict_image(face_frames)
                
                for bbox, age, gender in zip(face_bboxes, age_prediect, gender_predict):
                    x, y, w, h = bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    text = f'Age: {age}, Gender: {"Male" if gender == 0 else "Female"}'
                    cv2.putText(display_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0,0), 2)
                    
                    try:
                        recommendation.run_recommendation(age, gender)
                        rec_res = recommendation.get_recommendation()
                        cv2.putText(display_frame, rec_res, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)

                    except Exception as e:
                        print(f"Error in recommendation: {e}")


                # Display the frame
                cv2.imshow("Face Detection", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # 스레드가 종료될 때까지 기다림
        face_detection_run_thread.join()



if __name__ == "__main__":


    main()
