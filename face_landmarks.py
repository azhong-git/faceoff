import numpy as np
def convert_landmarks(landmarks_a, landmark_type_a, landmark_type_b):
    if landmark_type_b == 'muct' and landmark_type_a == 'muct_clmtools':
        landmarks_b = []
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
    elif landmark_type_b == 'muct' and landmark_type_a == 'lfpw':
        landmarks_b = np.ones((76 ,2)) * (-1)
        # landmark_b_index: landmark_a_index
        mapping = {27: 9, 28: 13, 29: 11, 30: 14, 31: 17,
                   32: 10, 33: 15, 34: 12, 35: 16, 36: 18,
                   39: 19, 43: 20, 67: 21, 41: 22,
                   48: 23, 54: 24, 51: 25, 57: 28, 64: 26, 61: 27,
                   7: 35}
        for i in mapping.keys():
            landmarks_b[i] = landmarks_a[mapping[i]-1]
        return np.array(landmarks_b)
    elif landmark_type_b == 'muct' and landmark_type_a == 'multipie':
        landmarks_b = np.ones((76 ,2)) * (-1)
        if len(landmarks_a) == 68:
            mapping = {0: 0, 7: 8, 13: 16,
                       27: 36, 29: 39, 68: 37, 69: 38, 70: 40, 71: 41,
                       34: 42, 32: 45, 73: 43, 72: 44, 74: 47, 75: 46,
                       67: 30, 41: 33,
                       48: 48, 54: 54, 51: 51, 57: 57, 64: 62, 61: 66}
        elif len(landmarks_a) == 66:
            mapping = {0: 0, 7: 8, 13: 16,
                       27: 36, 29: 39, 68: 37, 69: 38, 70: 40, 71: 41,
                       34: 42, 32: 45, 73: 43, 72: 44, 74: 47, 75: 46,
                       67: 30, 41: 33,
                       48: 48, 54: 54, 51: 51, 57: 57, 64: 61, 61: 64}
        else:
            assert False, 'should only accept multipie labels with 66 or 68'
        for i in mapping.keys():
            landmarks_b[i] = landmarks_a[mapping[i]]
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
    elif landmark_type == 'muct':
        landmark_index_dict['face'] = range(76)
        landmark_index_dict['left_eye']  = [27, 28, 29, 30, 31, 68, 69, 70, 71]
        landmark_index_dict['right_eye'] = [32, 33, 34, 35, 36, 72, 73, 74, 75]
        landmark_index_dict['mouth'] = range(48, 67)
    elif landmark_type == 'lfpw':
        landmark_index_dict['face'] = range(35)
        landmark_index_dict['left_eye']  = [8, 10, 12, 13, 16]
        landmark_index_dict['right_eye']  = [9, 11, 14, 15, 17]
        landmark_index_dict['mouth']  = [22, 23, 24, 25, 26, 27]
    elif landmark_type == 'multipie':
        landmark_index_dict['face'] = range(68)
        landmark_index_dict['left_eye'] = [36, 37, 38, 39, 40, 41]
        landmark_index_dict['right_eye'] = [42, 43, 44, 45, 46, 47]
        landmark_index_dict['mouth'] = [48, 51, 54, 57, 62, 66]
    else:
        assert False, 'landmark type {} not supported'.format(landmark_type)
    return landmark_index_dict

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
