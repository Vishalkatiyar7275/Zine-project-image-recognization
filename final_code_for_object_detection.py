

import cv2
import numpy as np

# define color ranges for the shapes
colors = {
    'red': [(0, 50, 50), (50, 255, 255)],
    'green': [(50, 50, 50), (100, 255, 255)],
    'blue': [(110, 50, 50), (150, 255, 255)]
}

# define shape detection functionimport cv2


def detect_shape(cnt):
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        x,y,w,h = cv2.boundingRect(approx)
        aspect_ratio = float(w)/h
        if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
            return "Square"
        
    elif len(approx) > 4:
        return "Circle"
    else:
        return None

# define video source
cap = cv2.VideoCapture(0)

while True:
    # read frame from video source
    ret, frame = cap.read()

    # convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # loop over color ranges and find contours of shapes
    for color in colors:
        lower, upper = colors[color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # loop over contours and detect shapes
        for cnt in contours:
            shape = detect_shape(cnt)
            if shape:
                # draw bounding box around shape
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # print shape and color to terminal
                print("Detected {} {} object".format(color, shape))

    # show frame with detected shapes
    cv2.imshow('frame', frame)

    # exit program on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release video source and destroy all windows
cap.release()
cv2.destroyAllWindows()
