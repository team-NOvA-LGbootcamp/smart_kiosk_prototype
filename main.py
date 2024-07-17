# main.py
import cv2

import threading

from face_detection import FaceDetection
from model_operation import ModelOperation
from recommendation import ProductRecommendationAlgorithm

def main():

    face_detection = FaceDetection(camera_index=0)
    face_detection_run_thread = threading.Thread(target=face_detection.run)
    face_detection_run_thread.start()

    product_recommendation_algorithm = ProductRecommendationAlgorithm(model_path="path/to/product_recommendation_model")

    try:
        while True:
            
            org_frame = face_detection.get_org_frame()
            face_frames = face_detection.get_face_frames()
            face_bboxes = face_detection.get_face_bboxes()
            
            if org_frame is not None:
                # Convert the frame back to BGR for OpenCV display
                display_frame = cv2.cvtColor(org_frame, cv2.COLOR_RGB2BGR)
                
                # Draw bounding boxes
                for bbox in face_bboxes:
                    x, y, w, h = bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
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
