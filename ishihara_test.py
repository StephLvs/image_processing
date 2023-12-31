import cv2
import matplotlib.pyplot as plt

# Read the image in colour (RGB)
img = cv2.imread('6.jpg', 1)
# Convert the image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# Show image
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()