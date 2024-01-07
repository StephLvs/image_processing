# Overview
This project integrates  an advanced AI tools: OpenAI ChatGPT-4.  
This image processing project consists of using computer vision techniques to enhance the visibility of Ishihara plate numbers making it easy for individuals with red-green colour vision defect. This README informs how AI technology have contributed to the project.
# Highlighting
In this section, I've used ChatGPT-4 to find the values of the different hues i needed to create masks. The masks are used to highlight the red and green colour.

## Chat GPT prompt
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

## Chat GPT prompt
```
I want to read all the images in the plates folder and output the images one by one.
```
### Expected Functionality
After applying the masks on one plate, I've created a folder with multiple images of the Ishihara plates to test the mask.   
I expected the masks to be applied on each plate and to view it one by one.
### Actual Functionality
The module functions as intended. Thanks to this module, I can see if the range of the masks correponds to the different hues of the Ishihara plates and adjust it if needed.

# Black and white

## Chat GPT prompt
```
```
### Response
### Expected Functionality
### Actual Functionality

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
