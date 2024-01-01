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
    hsv_refined = cv2.cvtColor(img_refined, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Apply masks
    #mask_red = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([20, 255, 255]))
    #mask_green = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([120, 255, 255]))

    mask_red_refined = cv2.inRange(hsv_refined, np.array([0, 100, 100]), np.array([20, 255, 255]))
    mask_red2_refined = cv2.inRange(hsv_refined, np.array([160, 100, 100]), np.array([180, 255, 255]))

    mask_green_refined = cv2.inRange(hsv_refined, np.array([20, 50, 50]), np.array([120, 255, 255]))
    #mask_green_refined = cv2.inRange(hsv_refined, np.array([25, 60, 60]), np.array([120, 255, 255]))
    mask_green2_refined = cv2.inRange(hsv_refined, np.array([20, 30, 30]), np.array([40, 255, 255]))

    # Combine masks
    red_mask = cv2.bitwise_or(mask_red_refined, mask_red2_refined)
    green_mask = cv2.bitwise_or(mask_green_refined, mask_green2_refined)

    # Outputs
    #output_red = cv2.bitwise_and(img, img, mask=mask_red)
    #output_green = cv2.bitwise_and(img, img, mask=mask_green)
    output_red_refined = cv2.bitwise_and(img_refined, img_refined, mask=mask_red_refined)
    red_refined = cv2.bitwise_and(img_refined, img_refined, mask=red_mask)

    output_green_refined = cv2.bitwise_and(img_refined, img_refined, mask=mask_green_refined)
    output_green2_refined = cv2.bitwise_and(img_refined, img_refined, mask=mask_green2_refined)
    green_refined = cv2.bitwise_and(img_refined, img_refined, mask= green_mask)
    

    # Show images
    #combined_refined = np.hstack((img, red_refined, green_refined ))
    combined_refined = np.hstack((img, output_green_refined, green_refined ))
    #combined = np.hstack((output_red, output_green))

    #cv2.imshow(f'{file}', combined)
    #cv2.waitKey(0)
    cv2.imshow(f'{file}', combined_refined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

