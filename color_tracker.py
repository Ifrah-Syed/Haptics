import cv2
import numpy as np

vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    resized_img = cv2.resize(frame, (600, 600))
    # resized_img=image
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)q

    lower = np.array([253])
    upper = np.array([255])
    # define blue color range
    # light_blue = np.array([110, 50, 50])
    # dark_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    # mask = cv2.inRange(hsv, light_blue, dark_blue)
    mask = cv2.inRange(gray, lower, upper)

    # Bitwise-AND mask and original image
    # output = cv2.bitwise_and(image, image, mask=mask)
    output = cv2.bitwise_and(gray, gray, mask=mask)
    kernel = np.ones((3, 3), np.float32) / 9
    mask = cv2.filter2D(mask, -1, kernel)
    # cv2.imshow("Color Detected", np.hstack((resized_img,output)))
    cv2.imshow("Mask", np.hstack((output, mask)))

    # Display the resulting frame
    #cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# After the loop release the cap object
vid.release()
#image = cv2.imread("1a.jpg")
# Convert BGR to HSV

"""""
resized_img=cv2.resize(image,(600,600))
#resized_img=image
gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


lower = np.array([200])
upper = np.array([255])
# define blue color range
#light_blue = np.array([110, 50, 50])
#dark_blue = np.array([130, 255, 255])

# Threshold the HSV image to get only blue colors
#mask = cv2.inRange(hsv, light_blue, dark_blue)
mask = cv2.inRange(gray, lower, upper)

# Bitwise-AND mask and original image
#output = cv2.bitwise_and(image, image, mask=mask)
output = cv2.bitwise_and(gray, gray, mask=mask)
kernel = np.ones((3,3),np.float32)/9
mask = cv2.filter2D(mask,-1,kernel)
"""

#cv2.imshow("Color Detected", np.hstack((resized_img,output)))
cv2.imshow("Mask", np.hstack((output, mask)))
cv2.waitKey(0)
cv2.destroyAllWindows()
