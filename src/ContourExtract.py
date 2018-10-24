import cv2 as cv
import numpy as np
import math

# Read Image and Resize to 1/4*1/4
imgorg = cv.imread(r"J:\\2552\\IMG_2552.JPG")
imgsub = cv.resize(imgorg, None, fx=0.25, fy=0.25, interpolation=cv.INTER_CUBIC)
imgsubgray = cv.cvtColor(imgsub, cv.COLOR_BGR2GRAY)

# Show Image...
# cv2.imshow("image", imgorg)
# cv2.waitKey(0)

# k-means clustering
Z = imgsub.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
K = 3
ret,label,center=cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((imgsub.shape))
imgcls = label.reshape(imgsub.shape[0], imgsub.shape[1])

# label names for memory
labelnames = ["Outter", "Shadow", "Oral"]
print(np.mean(center, axis=1))

# binary image, outter,oral=>0, shadow=>255
imgclsbin = np.uint8(imgcls)
imgclsbin[np.where(imgcls==1)] = 255
imgclsbin[np.where(imgcls==2)] = 0

# morphological operation - closing
kernel = np.ones((50, 50), np.uint8)
closing = cv.morphologyEx(imgclsbin, cv.MORPH_CLOSE, kernel)
#cv.imshow("closing", closing)

kernel = np.ones((5,5), np.uint8)
opening = cv.morphologyEx(imgclsbin, cv.MORPH_OPEN, kernel)
#cv.imshow("opening", opening)

cv.imshow("class binary", imgclsbin)
#cv.waitKey(0)

# find contours
inimg = np.uint8(closing)
im2, contours, hierarchy = cv.findContours(inimg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# contour
max_area = 0.0
max_cnt = contours[0]
sec_area = 0.0
sec_cnt = contours[0]
for cnt in contours:
    area = cv.contourArea(cnt)
    if area > max_area:
        if max_area > sec_area:
            sec_area = max_area
            sec_cnt = max_cnt
        max_area = area
        max_cnt = cnt
    elif area > sec_area:
        sec_area = area
        sec_cnt = cnt

cv.drawContours(imgsub, contours, -1, (0,255,0), 3)

x,y,w,h = cv.boundingRect(sec_cnt)
cv.rectangle(imgsub,(x,y),(x+w,y+h),(255,0,0),2)

rect = cv.minAreaRect(sec_cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(imgsub,[box],0,(0,0,255),2)

ellipse = cv.fitEllipse(sec_cnt)
cv.ellipse(imgsub,ellipse,(128,128,0),2)

cv.imshow("binary cluster image", imgsub)
cv.waitKey(0)

cv.destroyAllWindows()
