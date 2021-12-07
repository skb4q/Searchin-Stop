# Sooriya Bounyalith
# Image Segmentation for Stop Signs

import cv2
import numpy
import matplotlib
from matplotlib import pyplot as plt
import sys
import time


def nothing(x):
    pass


while True:
    # Will be used for looping for the contour section
    loop = 0

    # Display the image in a window
    img = cv2.imread('Stop Sign 8.jpg')

    # Use HSV to create a mask to filter out all the colors to produce mainly red
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create the lower and upper bound arrays with numpy
    lowerB = numpy.array([0, 88, 81])
    upperB = numpy.array([190, 255, 255])

    # Create mask to change all the red in the picture to white on a separate window
    RtoW = cv2.inRange(hsv, lowerB, upperB)

    # Used in a separate window to display only the red in an image
    detectRed = cv2.bitwise_and(img, img, mask=RtoW)

    # This section of code will be used to detect all of the contours in an image and
    # determine if something has 8 sides, as a stop sign would as an octagonal shape

    # Detects the contours of any shape detected with red in an image
    contours, _ = cv2.findContours(RtoW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loops over a few times in the case of stop signs being further away from the capture device
    # and allowing for more contours to be picked up
    while loop < 2:
        if loop == 0:
            contValue = 0.025
        elif loop == 1:
            contValue = 0.015

        for count in contours:
            # Finds the number of contours for a closed shape in an image
            approx = cv2.approxPolyDP(count, contValue* cv2.arcLength(count, True), True)

            # If the number of contours in a shape detected in the image is 8, it will interpret it as technically
            # being an octagon, which is the shape of a stop sign. Uncomment to see the actual contour.
            # cv2.drawContours(img, [approx], 0, (0, 0, 0), 2)

            # Draws a square over an object and prints text to establish something as being found as a stop sign
            if len(approx) == 8:
                # Get the coordinates along with the width and height value of the detected object
                x, y, w, h = cv2.boundingRect(approx)

                # Calculate the aspect ratio based on the width and height
                aspRatio = float(w)/h

                # Ensures that no tiny shapes that make it past the color filter will be picked up and
                # detected as a stop sign
                if w >= 30 and h >= 30:
                    # Aspect ratio values should be just enough to detect most 8-sided shapes
                    if 0.75 <= aspRatio <= 1.25:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(img, "Stop Sign", (x - 15, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))

        loop += 1

    # Displays all the windows to show if there is a stop sign within the image
    cv2.imshow('Stop Sign', img)
    cv2.imshow('Red to White', RtoW)

    break

cv2.waitKey(0)
cv2.destroyAllWindows()
sys.exit(0)
