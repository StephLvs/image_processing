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

## Chat GPT prompt
```
```
### Response
### Expected Functionality
### Actual Functionality

## Chat GPT prompt
```
```
### Response
### Expected Functionality
