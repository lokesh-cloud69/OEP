class MouthDetector:
    def detect_talking(self, face_landmarks, img_w, img_h, threshold=5):
        top_lip = face_landmarks[13]
        bottom_lip = face_landmarks[14]
        
        top_lip_y = int(top_lip.y * img_h)
        bottom_lip_y = int(bottom_lip.y * img_h)
        lip_distance = bottom_lip_y - top_lip_y
        
        is_talking = lip_distance > threshold
        return is_talking, lip_distance
