import os
import cv2
import numpy as np

path = '/disk/multi-pie/Multi-Pie/data/session01/multiview/003/01/08_1'
for filename in os.listdir(path):
    full_filename = os.path.join(path, filename)
    image = cv2.imread(full_filename)
    image = np.flipud(image)
    cv2.imwrite(full_filename, image)
