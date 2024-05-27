def get_look_direction(landmarks):
    nose_tip = landmarks['nose_tip'][2]  # Middle of the nose tip
    left_eye = landmarks['left_eye'][0]
    right_eye = landmarks['right_eye'][3]
    mid_eye = (left_eye[0] + right_eye[0]) / 2

    if abs(nose_tip[0] - mid_eye) < 5:
        return "Looking Straight"
    elif nose_tip[0] < mid_eye:
        return "Looking Left"
    else:
        return "Looking Right"
