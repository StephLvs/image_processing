import numpy as np
import cv2
import os

plates_folder = 'plates'
binary_folder = 'binary_plates'
files = os.listdir(plates_folder)

plate_images = []
binary_images = []
binary_file = []
for f in files:
    plate_img = cv2.imread(os.path.join(plates_folder, f), 0)
    binary_img = cv2.imread(os.path.join(binary_folder, f), 0)
    if plate_img is not None and binary_img is not None:
        plate_images.append(plate_img)
        binary_images.append(binary_img)
        binary_file.append(f)

# ORB detector
orb = cv2.ORB_create()

# Detection function
def detect_and_describe(image):
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors
# Match live plate to plate image
def match_plates(frame, plate_images):
    frame_keypoints, frame_descriptors = detect_and_describe(frame)
    best_matches = None
    best_image = None
    max_matches = 20
    best_idx = -1  # Initialize a variable to keep track of the best image index

    for idx, plate_img in enumerate(plate_images):
        plate_keypoints, plate_descriptors = detect_and_describe(plate_img)
        # Brute Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(frame_descriptors, plate_descriptors)

        # Update if this image has more matches
        if len(matches) > max_matches:
            max_matches = len(matches)
            best_matches = matches
            best_image = plate_img
            best_idx = idx  # Update the best index

    return best_image, best_matches, best_idx 


# ID of the camera
cam = cv2.VideoCapture(0)

while(True):
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect circles in the frame
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=4, minDist=1000, param1=80, param2=40, minRadius=60, maxRadius=200)
    if circles is not None:
        for i in circles[0,:]:
            i = list(map(int, i))
            #cv2.circle(frame,(i[0],i[1]),i[2]+20,(0,255,0),2)

            matched_image, matches, matched_idx = match_plates(gray, plate_images)

            if matched_image is not None and len(matches) > 1 and matched_idx is not None:
                superimpose_binary = binary_images[matched_idx]
                binary_path = binary_file[matched_idx]
                # Extract keypoint coordinates
                frame_keypoints, _ = detect_and_describe(gray)
                plate_keypoints, _ = detect_and_describe(matched_image)

                src_pts = np.float32([frame_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([plate_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Find homography and superimpose
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                height, width = frame.shape[:2]
                output = cv2.warpPerspective(superimpose_binary, M, (width, height))
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            

                # Create a mask from the circle
                mask = np.zeros(gray.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (i[0], i[1]), i[2]+ 20, 255, -1)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # Resize the binary image
                resized_binary = cv2.resize(superimpose_binary, (mask.shape[1]//2, mask.shape[0]//2))

                if len(resized_binary.shape) == 2:
                    resized_binary = cv2.cvtColor(resized_binary, cv2.COLOR_GRAY2BGR)

                # Calculate the position to place the resized binary image
                y_offset = i[1] - resized_binary.shape[0] // 2
                x_offset = i[0] - resized_binary.shape[1] // 2

                y_offset = max(0, min(y_offset, mask.shape[0] - resized_binary.shape[0]))
                x_offset = max(0, min(x_offset, mask.shape[1] - resized_binary.shape[1]))

                # Place the resized binary image onto the mask
                mask[y_offset:y_offset+resized_binary.shape[0], x_offset:x_offset+resized_binary.shape[1]] = resized_binary

                # Superimpose the binary image onto the circle
                masked_frame = cv2.bitwise_and(frame, 255 - mask)
                superimposed_image = cv2.add(masked_frame, mask)

                text_position = (i[0] + i[2] + 100, i[1] + 200)
                number = binary_path.split("_")[-1].split(".")[0]
                superimposed_image = cv2.putText(superimposed_image, f'Number {number}', text_position, 4, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                combine = np.hstack((frame, superimposed_image))
                cv2.imshow('Cheat Ishihara Test', combine)  

    if cv2.waitKey(5) == 27:
        break


cv2.destroyAllWindows()
cam.release()