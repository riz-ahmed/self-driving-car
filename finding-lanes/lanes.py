import cv2
import numpy as np
import math

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   # convert RGB to gray-scale
    blur = cv2.GaussianBlur(gray, (5,5), 0)         # image smoothening using GaussianBlur with 5,5 grid and 0 deviation
    canny = cv2.Canny(blur, 50, 150)                # apply canny to find the edges
    return canny

def region_of_interest(image):
    height = image.shape[0]                         # defien the height of the image in pixels
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])    # this polygon (triangle) represents the area of region_of_interest
    mask = np.zeros_like(image)                     # will create a image array of all black pixles
    cv2.fillPoly(mask, polygons, 255)               # fill black image with traingle of all white pixels
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image                                     # the changes are already applied on mask as this is a np array

def display_lines(image, lines):
    lane_image = np.zeros_like(image)                 # create a copy of the lane lane_image filled with all black pixles
    if lines is not None:                             # select onyl the non empty lines from the houges HoughLinesP
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)              # reshape the returned HoughLinesP to a 1D 4 element array and unpack for each of the point coordinates
            cv2.line(lane_image, (x1,y1), (x2,y2), (255,0,0), 10)   # draw these lines on array formed using lane_image
    return lane_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(1/2))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []                                # empty list for left lines
    right_fit = []                              # empty list for right lines
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)              # reshape the returned HoughLinesP to a 1D 4 element array and unpack for each of the point coordinates
        line_parameters = np.polyfit((x1, x2),(y1, y2),1)   # return slope and intercept for a straight line
        slope = line_parameters[0]
        intercept = line_parameters[1]
        if slope < 0:
            # negative slope corresponds to left side of the lines in the region_of_interest
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

        # take the avaerage of these slopes and intercepts
    left_fit_avg = np.average(left_fit, axis=0)     # axis=0 indicates vertical average along the columns
    right_fit_avg = np.average(right_fit, axis=0)

    # call a function make_coordinates() to plot these lines using some coordinates
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)

    # return the line coordinates returned by the called function make_coordinates()
    return np.array([left_line, right_line])

# image = cv2.imread('test_image.jpg')          # read the image and returns the pixeled array
# lane_image = np.copy(image)                    # make a copy for gray-scale
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('result',combined_image)                        # display the test_image in blur
# cv2.waitKey(0)                                  # displays the image indefinitely

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()                       # this method returns the vedio into various frames
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result',combined_image)                        # display the test_image in blur
    cv2.waitKey(1)                                             # 1 ms of wait time between each of the frames
