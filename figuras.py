
import cv2 as cv

img = cv.imread('man1.jpg', 1)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
ubb=(0, 100, 100)
uba=(10, 255, 255)
ubb2=(170, 100, 100)
uba2=(180, 255, 255)
ubb3=(40,100,100)
uba3=(80,255,255)

mask1 = cv.inRange(hsv, ubb, uba)
mask2 = cv.inRange(hsv, ubb2, uba2)
mask3 = cv.inRange(hsv, ubb3, uba3)
mask = mask1 + mask2


res = cv.bitwise_and(img, img, mask=mask)
cv.imshow('mask', mask)
cv.imshow('hsv', hsv)
cv.imshow('res', res)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()




