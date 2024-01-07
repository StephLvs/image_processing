import numpy as np
import cv2
import os

# Initialize the ORB detector
orb = cv2.ORB_create()

folder_path = 'plates'
binary_folder = 'binary_plates'
files = os.listdir(folder_path)

plate_images = []
binary_images = []
for f in files:
    plate_img = cv2.imread(os.path.join(folder_path, f), 0)
    binary_img = cv2.imread(os.path.join(binary_folder, f), 0)
    if plate_img is not None and binary_img is not None:
        plate_images.append(plate_img)
        binary_images.append(binary_img)

# Feature detection function
def detect_and_describe(image):
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# Function to match frame to still images
def match_frame_to_still(frame, plate_images):
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

    return best_image, best_matches, best_idx  # Return the best index


# takes id of the camera
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Match frame to still images
    matched_image, matches, matched_idx = match_frame_to_still(frame_gray, plate_images)

   # If a match is found and is significant
    if matched_image is not None and len(matches) > 1 and matched_idx is not None:
        binary_img_to_superimpose = binary_images[matched_idx]

        # Extract keypoint coordinates
        frame_keypoints, _ = detect_and_describe(frame_gray)
        plate_keypoints, _ = detect_and_describe(matched_image)

        src_pts = np.float32([frame_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([plate_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography and superimpose
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Warp the binary image using the homography matrix
        h, w = frame.shape[:2]
        im_out = cv2.warpPerspective(binary_img_to_superimpose, M, (w, h))

        # Convert im_out to a 3 channel image
        im_out_3_channel = cv2.cvtColor(im_out, cv2.COLOR_GRAY2BGR)

        # Create a mask from the warped binary image
        mask = cv2.cvtColor(im_out, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel mask
        mask[mask > 0] = 1  # Create a binary mask

        # Superimpose only on the matched area
        superimposed_image = cv2.multiply(frame, (1-mask))  # Remove the matched area from the original frame
        superimposed_image += cv2.multiply(im_out_3_channel, mask)  # Add the binary image only in the matched area

        cv2.imshow("Superimposed Image", superimposed_image)
    else:
        cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) == 27:  # Escape key
        break




cv2.destroyAllWindows()
cam.release()
