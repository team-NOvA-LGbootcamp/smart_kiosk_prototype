import cv2
import mediapipe as mp


class FaceDetection:
    def __init__(
        self,
        camera_index=0,
        image_size=(640, 480),
    ):
        # Initialize the camera capture
        self.cap = cv2.VideoCapture(camera_index)
        self.cam_w, self.cam_h = image_size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initialize the Mediapipe face detection model
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        self.face_frames = []
        self.face_bboxes = []

        self.frame = None  # Initialize self.frame

    def detect_faces(self, frame):
        # Perform face detection
        f_frames = []
        f_bboxes = []
        results = self.mp_face_detection.process(frame)

        # If faces are detected
        if results.detections:
            for detection in results.detections:
                mp_bbox = detection.location_data.relative_bounding_box
                if (
                    mp_bbox.xmin < 0
                    or mp_bbox.ymin < 0
                    or mp_bbox.xmin + mp_bbox.width >= self.cam_w
                    or mp_bbox.ymin + mp_bbox.height >= self.cam_h
                ):
                    break
                bbox = (
                    int(mp_bbox.xmin * self.cam_w),
                    int(mp_bbox.ymin * self.cam_h),
                    int(mp_bbox.width * self.cam_w),
                    int(mp_bbox.height * self.cam_h),
                )

                x, y, w, h = bbox

                f_bboxes.append(bbox)
                f_frames.append(frame[y : y + h, x : x + w])

        return f_frames, f_bboxes

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.face_frames, self.face_bboxes = self.detect_faces(self.frame)

        self.cap.release()

    def get_org_frame(self):
        return self.frame

    def get_face_frames(self):
        return self.face_frames

    def get_face_bboxes(self):
        return self.face_bboxes
