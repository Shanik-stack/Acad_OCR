import cv2
import os

folder_path_1 = f"extracted_diagram/"

for j in[folder_path_1]:
    for i in os.listdir(j):
        # print(i)
        img = cv2.imread(j+i)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(j+i,gray_img)
