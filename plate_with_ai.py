import cv2
import os
import easyocr
import imutils

path = os.path.dirname(os.path.realpath(__file__))
img = cv2.imread(os.path.join(path,"images","img1.jpg"))

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plates = cv2.CascadeClassifier(os.path.join(path, "russian_plate_number.xml"))

results = plates.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)

for (x, y, w, h) in results:
    y1 = x
    x1 = y
    y2 = x + w
    x2 = y + h
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), thickness=3)


crop = gray[x1:x2, y1:y2]

edges = cv2.Canny(crop, 30, 200)
contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

hull_list = []
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    hull_list.append(hull)

hull_list = sorted(hull_list, key=cv2.contourArea,reverse=True)
bounding_rect = cv2.boundingRect(hull_list[0])
print(f"bounding_rect is {bounding_rect}")

# cv2.drawContours(crop, contours, -1, (0, 255, 0), 1)
cv2.rectangle(crop, (bounding_rect[0], bounding_rect[1]), (bounding_rect[0]+bounding_rect[2], bounding_rect[1]+bounding_rect[3]), (0,255,0),2)
crop = crop[bounding_rect[1]:bounding_rect[1]+bounding_rect[3],bounding_rect[0]:bounding_rect[0]+bounding_rect[2]]
print(contours)
(thresh, im_bw) = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Crop",im_bw)

text = easyocr.Reader(['en'])
text = text.readtext(im_bw)
res = text[0][-2]

print('\n')
print(res)

cv2.imshow("Result",img)
cv2.waitKey(0)
