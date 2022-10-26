import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as plt
import os

path = os.path.dirname(os.path.realpath(__file__))
img = cv2.imread(os.path.join(path,"images","img3.jpg"))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_filter = cv2.bilateralFilter(gray, 11, 15, 15)
edges = cv2.Canny(img_filter, 30, 200)
contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

pos = None # probable plate position
for contour in contours:
    approx = cv2.approxPolyDP(contour, 8, True) # second value is square probability, more=
    if len(approx) == 4:
        pos = approx
        break
print(f"pos is {pos}")

mask = np.zeros(gray.shape, dtype=np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
crop = gray[x1:x2, y1:y2]

text = easyocr.Reader(['en'])
text = text.readtext(crop)
res = text[0][-2]
final_image = cv2.putText(img, res, (x1, y2 + 60), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
final_image = cv2.rectangle(img, (x1, x2),(y1,y2), (0,255,0),2)

print('\n')
print(res)

plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.show()

# cv2.imshow("image1",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()