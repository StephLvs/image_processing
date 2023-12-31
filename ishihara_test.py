import cv2
import os
import numpy as np

folder_path = 'plates'
files= os.listdir(folder_path)
kernel = np.ones((5, 5), np.uint8)

for file in files:
    # Read the image in colour (RGB)
    plate_img = os.path.join(folder_path, file)
    img = cv2.imread(plate_img, 1)

    img_refined = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_refined = cv2.morphologyEx(img_refined, cv2.MORPH_CLOSE, kernel)

    # Convert the image to HSV
    hsv = cv2.cvtColor(img_refined, cv2.COLOR_BGR2HSV)

    # Apply masks
    mask_red = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([20, 255, 255]))
    mask_green = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([120, 255, 255]))


    output_red = cv2.bitwise_and(img_refined, img_refined, mask=mask_red)
    output_green = cv2.bitwise_and(img_refined, img_refined, mask=mask_green)

    # Show images
    combined = np.hstack((img, output_red, output_green))

    cv2.imshow(f'{file}', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

