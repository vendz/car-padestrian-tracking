import cv2

# read the image file
video = cv2.VideoCapture ('Tesla Autopilot Dashcam.mp4')

# pre-trained classifiers
car_classifier_file = 'car_detector.xml'
padestrian_classifier_file = 'padestrians.xml'

# create classifiers
car_tracker = cv2.CascadeClassifier(car_classifier_file)
padestrian_tracker = cv2.CascadeClassifier(padestrian_classifier_file)


while True:

    # read the current frame
    (read_successful, frame) = video.read()

    # safe coding.
    if read_successful:
        # changing color to greyscale
        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    # detect cars
    cars = car_tracker.detectMultiScale(frame)
    padestrians = padestrian_tracker.detectMultiScale(frame)

    # draw square around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # draw square around the padestrians
    for (x, y, w, h) in padestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # this will display the image
    cv2.imshow('vendz car and padestrian tracking', frame)

    # it don't autoclose
    # it will wait until you press a key to close the window
    key = cv2.waitKey(1)

    # Stop is Q key is pressed
    if key == 81 or key == 113:  # here 113 is for 'q' and 81 is for 'Q'
        break