import cv2
import numpy as np

class PoseEstimator:
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             
            (0.0, -330.0, -65.0),        
            (-225.0, 170.0, -135.0),     
            (225.0, 170.0, -135.0),      
            (-150.0, -150.0, -125.0),    
            (150.0, -150.0, -125.0)      
        ], dtype=np.float64)

    def estimate_pose(self, face_landmarks, img_w, img_h):
        face_2d = []
        landmark_indices = [1, 152, 33, 263, 61, 291] 
        nose_2d, nose_3d = None, None
        
        for idx in landmark_indices:
            lm = face_landmarks[idx] 
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            if idx == 1:
                nose_2d = (x, y)
                nose_3d = (x, y, lm.z * 3000)
                
        face_2d = np.array(face_2d, dtype=np.float64)
        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        success, rot_vec, trans_vec = cv2.solvePnP(self.model_points, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        pitch = angles[0] * 360
        yaw = angles[1] * 360
        roll = angles[2] * 360
        
        direction = "Forward"
        if yaw < -10: direction = "Looking Left"
        elif yaw > 10: direction = "Looking Right"
        elif pitch < -10: direction = "Looking Down"
        elif pitch > 10: direction = "Looking Up"
            
        return direction, pitch, yaw, roll, nose_2d, nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix
