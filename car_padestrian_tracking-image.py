import cv2

# our image
img_file = 'cars1.jpeg'

# pre-trained car classifier
classifier_file = 'car_detector.xml'
padestrian_classifier_file = 'padestrians.xml'

# read the image file
img = cv2.imread(img_file)

# create a classifier
car_tracker = cv2.CascadeClassifier(classifier_file)
padestrian_tracker = cv2.CascadeClassifier(padestrian_classifier_file)

# changing color to greyscale
greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect cars
cars = car_tracker.detectMultiScale(greyscale_img)
padestrians = padestrian_tracker.detectMultiScale(greyscale_img)

# draw square around the cars
for (x ,y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
for (x, y, w, h) in padestrians:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# this will display the image
cv2.imshow('vendz car and padestrian tracking', img)

# it don't autoclose
# it will wait until you press a key to close the window
cv2.waitKey()