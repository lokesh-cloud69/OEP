import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import os
import urllib.request
import time

from pose_estimator import PoseEstimator
from mouth_detector import MouthDetector
from object_detector import ObjectDetector
from alert_logger import AlertLogger

def main():
    print("Initializing components...")
    pose_estimator = PoseEstimator()
    mouth_detector = MouthDetector()
    object_detector = ObjectDetector()
    alert_logger = AlertLogger()
    
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        print("Downloading MediaPipe Face Landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    
    base_options = BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=2
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 20.0 
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('proctor_session.mp4', fourcc, fps, (frame_width, frame_height))
    
    is_paused = False
    
    print("Starting webcam feed...")
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
            
        img_h, img_w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        person_count, phone_detected, obj_bboxes = object_detector.detect(image)
        
        alert_text = []
        if not is_paused:
            if alert_logger.track_state("Multiple Persons", person_count > 1, threshold_seconds=2):
                alert_text.append("WARNING: Multiple Persons Detected!")
            if alert_logger.track_state("Phone Detected", phone_detected, threshold_seconds=1):
                alert_text.append("WARNING: Mobile Phone Detected!")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = face_landmarker.detect(mp_image)
        
        faces_detected = 0
        if detection_result.face_landmarks:
            faces_detected = len(detection_result.face_landmarks)
            for face_landmarks in detection_result.face_landmarks:
                
                direction, pitch, yaw, roll, nose_2d, nose_3d, rvec, tvec, cam_matrix, dist_matrix = \
                    pose_estimator.estimate_pose(face_landmarks, img_w, img_h)
                
                is_looking_away = direction != "Forward"
                if not is_paused and alert_logger.track_state("Looking Away", is_looking_away, threshold_seconds=2):
                    alert_text.append(f"WARNING: Looking Away ({direction})")
                
                cv2.putText(image, f"Head Pose: {direction}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                if nose_2d:
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rvec, tvec, cam_matrix, dist_matrix)[0][0][0]
                    p2 = (int(p2[0]), int(p2[1]))
                    cv2.line(image, p1, p2, (255, 0, 0), 3)

                is_talking, lip_dist = mouth_detector.detect_talking(face_landmarks, img_w, img_h)
                if not is_paused and alert_logger.track_state("Talking", is_talking, threshold_seconds=2):
                    alert_text.append("WARNING: Talking Detected")
                
                cv2.putText(image, f"Talking: {is_talking} (Dist: {lip_dist})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                break
        
        if faces_detected == 0 and person_count == 0:
            if not is_paused and alert_logger.track_state("No Face", True, threshold_seconds=3):
                alert_text.append("WARNING: No Face Detected")

        for (x1, y1, x2, y2, label, conf) in obj_bboxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if is_paused:
            cv2.putText(image, "SYSTEM PAUSED", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        y_pos = 130 if is_paused else 90
        for alert in alert_text:
            cv2.putText(image, alert, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_pos += 30

        out_video.write(image)

        cv2.imshow('Exam Proctoring System', image)
        
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == ord('s') or key == ord('S'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, image)
            print(f"Screenshot saved: {filename}")
        elif key == ord('p') or key == ord('P'):
            is_paused = not is_paused
            state = "PAUSED" if is_paused else "RESUMED"
            print(f"Logging {state}")
            alert_logger.log_anomaly("System Event", f"Logging {state}")
            
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
