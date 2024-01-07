# Overview
This project integrates  an advanced AI tools: OpenAI ChatGPT-4.

This image processing project consists of using computer vision techniques to enhance the visibility of Ishihara plate numbers making it easy for individuals with red-green colour vision defect. This README informs how AI technology have contributed to the project.

# Highlighting
In this section, I've used ChatGPT-4 to find the values of the different hues i needed to create the masks and to read files from a folder to apply the masks onto it. The masks are used to highlight the red and green colour.

## Chat GPT prompt: Colors in HSV, OpenCV
```
What are the red hues in HSV colour space using OpenCV?   
What are the green hues in HSV colour space?   
What are the yellow hues in HSV?   
How to get gray colour in HSV?    
How to get brown colour in HSV?
```
### Expected Functionality
Accurately give the hues of the colours needed to create a green mask and a red mask.
### Actual Functionality
The module functions as intended.

## Chat GPT prompt: Read plates from folder
```
I want to read all the images in the plates folder and output the images one by one.
```
### Expected Functionality
After applying the masks on an image of a plate, I've created a folder with multiple images of the Ishihara plates to test the mask.   
I expected the masks to be applied on each plate and to view it one by one.
### Actual Functionality
The module functions as intended. Thanks to this module, I can see if the range of the masks correponds to the different hues of the Ishihara plates and adjust it if needed.

# Black and white
In this section, after having applied the red/ green colours masks to highlight the number. Chat GPT helped keep a copy of the binary version of the results.  

## Chat GPT prompt: Save the correct binary images to a folder
```
binary_red = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
binary_green = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)

# Save the binary image in the binary_folder
binary_img = os.path.join(binary_folder, file)
# Save the binary image that contains the least of white areas
cv2.imwrite(binary_img, binary_red)
```
### Expected Functionality
After succesfully masking the green and red colours, I wanted a binary version of the masks since the number on the plates appear clearer with just 2 uniform colours. The contrast between black and white is stronger than any other shades.  We have 2 binary images per image of the plate since I applied 2 different masks (red_mask and a green_mask).  I expect the binary_folder to contain the binary images where the number is white (so where there are less white areas). 
### Actual Functionality
The module functions as intended. 
```
# Calculate white area in both binary images
white_area_red = np.sum(binary_red == 255)
white_area_green = np.sum(binary_green == 255)

# Save only the binary image with less white area
if white_area_red < white_area_green:
    chosen_binary_img = binary_red
else:
    chosen_binary_img = binary_green

# Save the chosen binary image
binary_img_path = os.path.join(binary_folder, file)
cv2.imwrite(binary_img_path, chosen_binary_img)
```
# Live video
In this section, Chat GPT helped with the matching of the image of the plate to the actual image during a live feed. It then helped to superimpose the matched image onto the actual image.

## Chat GPT prompt: Matching and superimpostion
```
The plates during the live video should be matched to one of the still image of the plate.
We have to superimpose each still image on the frame to determine which plate and number it corresponds to.
```
### Expected Functionality
Give an idea of the functions to use to match an image to a live video frame and how to implement a superimposition.
### Actual Functionality
The module functions as intended. Chat GPT gives an explanation on how to use the ORB functionality of OpenCV, the function to get the keypoints of an image/ frame, the function to verify the matching using those keypoints and OpenCV's functions findHomogrpahy and warpPerpective for the superimposition.
I changed the values of 'max_matches' and MIN_MATCH_COUNT for a more accuarte matching.
```
# Feature detection function
def detect_and_describe(image):
    # Compute ORB key points and descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# Function to match frame to still images
def match_frame_to_still(frame, plate_images):
    frame_keypoints, frame_descriptors = detect_and_describe(frame)
    best_matches = None
    best_image = None
    max_matches = 20

    for plate_img in plate_images:
        plate_keypoints, plate_descriptors = detect_and_describe(plate_img)
        # Brute Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(frame_descriptors, plate_descriptors)
        
        # Update if this image has more matches
        if len(matches) > max_matches:
            max_matches = len(matches)
            best_matches = matches
            best_image = plate_img

    return best_image, best_matches

MIN_MATCH_COUNT = 1
# takes id of the camera
cam = cv2.VideoCapture(0)
while(True):
    ret, frame = cam.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Match frame to still images
    matched_image, matches = match_frame_to_still(frame_gray, plate_images)

    # If a match is found and is significant, superimpose the image
    if matched_image is not None and len(matches) > MIN_MATCH_COUNT:
        # Homography and superimposition logic goes here

        # Display the superimposed image
        cv2.imshow("Superimposed Image", superimposed_image)
    else:
        cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cam.release()
```

## Chat GPT prompt: Resizing of binary image onto the actual image during the live feed
```
How to overlay the resized binary image (which is smaller than the mask) into the middle of the mask.
```
```
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
            cv2.circle(frame,(i[0],i[1]),i[2]+20,(0,255,0),2)


            circle_roi = extract_circle_roi(gray, i)
            matched_image, matches, matched_idx = match_plates(gray, plate_images)
            if matched_image is not None and len(matches) > 1 and matched_idx is not None:
                binary_img_to_superimpose = binary_images[matched_idx]
                # Extract keypoint coordinates
                frame_keypoints, _ = detect_and_describe(gray)
                plate_keypoints, _ = detect_and_describe(matched_image)

                src_pts = np.float32([frame_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([plate_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Find homography and superimpose
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                height, width = frame.shape[:2]
                output = cv2.warpPerspective(binary_img_to_superimpose, M, (width, height))
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            

                # Create a mask from the circle
                mask = np.zeros(gray.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (i[0], i[1]), i[2]+ 20, 255, -1)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                resized_binary = cv2.resize(binary_img_to_superimpose, (mask.shape[1]-20, mask.shape[0]-20))

                # 2 to 3 channels
                if len(resized_binary.shape) == 2:
                    resized_binary = cv2.cvtColor(resized_binary, cv2.COLOR_GRAY2BGR)

                # Overlay the resized binary image onto the mask from the circle
                
                    
                # Superimpose the binary image onto the circle
                masked_frame = cv2.bitwise_and(frame, 255 - mask) # black circle 
                masked_out = cv2.bitwise_and(resized_binary, mask) # black frame
                superimposed_image = cv2.add(masked_frame, masked_out)

                combine = np.hstack((frame, superimposed_image))
                cv2.imshow('Cheat Ishihara Test', superimposed_image)  

    if cv2.waitKey(5) == 27:
        break


cv2.destroyAllWindows()
cam.release()
```
### Expected Functionality
After detecting the circle where the Ishihara plate lies on, I needed to resize the binary image to fit into the circle.
### Actual Functionality
The module functions as intended. Chat GPT calculated the position for embedding the resized binary image to center it within the circle.




