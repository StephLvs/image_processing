import cv2
import os
import numpy as np

folder_path = 'plates'
files= os.listdir(folder_path)
kernel = np.ones((15,15), np.uint8)

for file in files:
    # Read the image in colour (RGB)
    plate_img = os.path.join(folder_path, file)
    img = cv2.imread(plate_img, 1)

    img_refined = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_refined = cv2.morphologyEx(img_refined, cv2.MORPH_CLOSE, kernel)

    # Convert the image to HSV (hue (colour), saturation(grayish to vibrant, value (dark to light))
    hsv_refined = cv2.cvtColor(img_refined, cv2.COLOR_BGR2HSV)

    # Apply masks and combine masks
    mask_red_refined = cv2.inRange(hsv_refined, np.array([0, 80, 150]), np.array([20, 255, 255]))
    mask_red2_refined = cv2.inRange(hsv_refined, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask_red_refined, mask_red2_refined)

    mask_green_refined = cv2.inRange(hsv_refined, np.array([20, 20, 50]), np.array([120, 125, 255])) # average green
    mask_green2_refined = cv2.inRange(hsv_refined, np.array([20, 0, 20]), np.array([150, 50, 180])) # grayish dark green
    mask_green3_refined = cv2.inRange(hsv_refined, np.array([0, 20, 0]), np.array([20, 160, 135])) # brownish grayish light green
    green_mask = cv2.bitwise_or(mask_green_refined, mask_green2_refined)
    green_mask = cv2.bitwise_or(green_mask, mask_green3_refined)


    # Outputs
    output_red_refined = cv2.bitwise_and(img_refined, img_refined, mask=mask_red_refined)
    red_refined = cv2.bitwise_and(img_refined, img_refined, mask=red_mask)

    output_green_refined = cv2.bitwise_and(img_refined, img_refined, mask=mask_green_refined)
    output_green2_refined = cv2.bitwise_and(img_refined, img_refined, mask=mask_green2_refined)
    green_refined = cv2.bitwise_and(img_refined, img_refined, mask= green_mask)

    binary_red = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    binary_green = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)

    # Show images
    combined_refined = np.hstack((img, binary_red, binary_green))

    cv2.imshow(f'{file}', combined_refined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

