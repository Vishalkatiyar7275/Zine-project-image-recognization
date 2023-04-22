import cv2 as cv
import numpy as np

# define constants for barcode detection
BARCODE_WIDTH = 100  # in mm
STRIP_WIDTHS = [10, 20]  # in mm
BLACK_BORDER = 10  # in mm

# define video source
cap = cv.VideoCapture(0)

while True:
    # read frame from video source
    ret, frame = cap.read()

    # convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # apply binary thresholding to isolate black and white regions
    _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # find contours of white regions
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # loop over contours and find barcode
    for cnt in contours:
        # calculate contour area and perimeter
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)

        # calculate aspect ratio and check if it's within expected range
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
        expected_aspect_ratio = (BARCODE_WIDTH + 2 * BLACK_BORDER) / (4 * sum(STRIP_WIDTHS))
        if abs(aspect_ratio - expected_aspect_ratio) > 0.2:
            continue

        # check if the area is within expected range
        expected_area = BARCODE_WIDTH * (h - 2 * BLACK_BORDER)
        if abs(area - expected_area) > 0.2 * expected_area:
            continue

        # extract the barcode region from the frame
        barcode = gray[y + BLACK_BORDER:y + h - BLACK_BORDER, x:x + w]

        # resize barcode to fixed width and height
        barcode = cv.resize(barcode, (BARCODE_WIDTH, h - 2 * BLACK_BORDER), interpolation=cv.INTER_AREA)

        # convert barcode to binary image
        _, barcode = cv.threshold(barcode, 127, 255, cv.THRESH_BINARY_INV)

        # decode barcode and print result to terminal
        bits = ""
        for i in range(4):
            strip_width = STRIP_WIDTHS[i % 2]
            strip = barcode[:, i * strip_width:(i + 1) * strip_width]
            bit = 1 if np.mean(strip) < 127 else 0
            bits += str(bit)
        number = int(bits, 2)
        print("Detected barcode: {}".format(number))

        # draw bounding box around barcode
        cv.rectangle(frame, (x, y + BLACK_BORDER), (x + w, y + h - BLACK_BORDER), (0, 255, 0), 2)

    # show frame with detected barcode
    cv.imshow('frame', frame)

    # exit program on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# release video source and destroy all windows
cap.release()
cv.destroyAllWindows()
