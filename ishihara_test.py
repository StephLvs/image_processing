import cv2
import os
import numpy as np

plates_folder = 'plates'
binary_folder = 'binary_plates'

files= os.listdir(plates_folder)
kernel = np.ones((12,12), np.uint8)
image_dict = {}

for file in files:
    # Read the image in colour (RGB)
    plate_img = os.path.join(plates_folder, file)
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


    # Calculate white area in both binary images
    white_area_red = np.sum(binary_red == 255)
    white_area_green = np.sum(binary_green == 255)

    # Save only the binary image with less white area
    if white_area_red < white_area_green:
        chosen_binary_img = binary_red
    else:
        chosen_binary_img = binary_green

    # Save the chosen binary image
    binary_img = os.path.join(binary_folder, file)
    #chosen_binary_img = 255 - chosen_binary_img
    cv2.imwrite(binary_img, chosen_binary_img)

    # Update dictionary
    image_dict[plate_img] = binary_img


    # Show images
    combined_refined = np.hstack((img, binary_red, binary_green))

    cv2.imshow(f'{file}', combined_refined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(image_dict)
