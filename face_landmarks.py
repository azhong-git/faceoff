import numpy as np
def convert_landmarks(landmarks_a, landmark_type_a, landmark_type_b):
    landmarks_b = []
    if landmark_type_b == 'muct' and landmark_type_a == 'muct_clmtools':
        for i in range(0, 19):
            landmarks_b.append(landmarks_a[i])
        for j in range(2):
            landmarks_b.append(np.array([-1, -1]))
        for i in range(19, 23):
            landmarks_b.append(landmarks_a[i])
        for j in range(2):
            landmarks_b.append(np.array([-1, -1]))
        for i in range(23, 33):
            landmarks_b.append(landmarks_a[i])
        for j in range(1):
            landmarks_b.append(np.array([-1, -1]))
        for i in range(34, 41):
            landmarks_b.append(landmarks_a[i])
        for j in range(1):
            landmarks_b.append(np.array([-1, -1]))
        for i in range(42, 62):
            landmarks_b.append(landmarks_a[i])
        for j in range(1):
            landmarks_b.append(np.array([-1, -1]))
        for i in range(62, 71):
            landmarks_b.append(landmarks_a[i])
        return np.array(landmarks_b)
    else:
        assert False, 'conversion from {} to {} not supported'.format(landmark_type_a, landmark_type_b)

def get_landmark_index_dict(landmark_type):
    landmark_index_dict = {}
    if landmark_type  == 'muct_clmtools':
        landmark_index_dict['face'] = range(71)
        landmark_index_dict['left_eye']  = [23, 24, 25, 26, 27, 63, 64, 65, 66]
        landmark_index_dict['right_eye'] = [28, 29, 30, 31, 32, 67, 68, 69, 70]
        landmark_index_dict['mouth'] = range(44, 62)
        return landmark_index_dict
    elif landmark_type == 'muct':
        landmark_index_dict['face'] = range(76)
        landmark_index_dict['left_eye']  = [27, 28, 29, 30, 31, 68, 69, 70, 71]
        landmark_index_dict['right_eye'] = [32, 33, 34, 35, 36, 72, 73, 74, 75]
        landmark_index_dict['mouth'] = range(48, 67)
        return landmark_index_dict
    else:
        assert False, 'landmark type {} not supported'.format(landmark_type)

def flip_landmarks(landmarks, landmark_type):
    assert landmark_type == 'muct', 'only supporting muct landmark_type as of now'
    mappings = [(0, 14), (1, 13), (2, 12), (3, 11), (4, 10), (5, 9), (6, 8),
                (48, 54), (49, 53), (50, 52), (59, 55), (58, 56), (65, 63), (60, 62),
                (39, 43), (40, 42), (46, 47), (38, 44), (37, 45),
                (24, 18), (23, 17), (22, 16), (21, 15),
                (29, 34), (27, 32), (31, 36), (28, 33), (30, 35),
                (69, 73), (70, 74), (68, 72), (71, 75)]
    for ia, ib in mappings:
        temp = landmarks[ia].copy()
        landmarks[ia] = landmarks[ib]
        landmarks[ib] = temp
