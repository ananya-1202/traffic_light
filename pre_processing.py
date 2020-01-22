import cv2 
import numpy as np
import matplotlib.pyplot as plt 


#uploading the image

a='image.jpg'
img=cv2.imread(a)


#converting the image into HSV to extract color
  
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_red = np.array([170,120,70])
upper_red = np.array([180,255,255])
mask = cv2.inRange(img,lower_red,upper_red)


#Removing noise from the extracted image

element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
mask = cv2.erode(mask, element, iterations = 2)
mask = cv2.dilate(mask, element, iterations = 5)
mask = cv2.erode(mask, element)

#forming boundaries around the red region this is done to get the figure which in in shape of circle 
#since the traffic lights are not very big hence by changing the parameters of the drawContour function we can make the the traffic look like a complete circle and other objects wil have different shapes

ret, thresh = cv2.threshold(mask,50,255,0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img8=cv2.imread(a)
img8=0*img8
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    cv2.drawContours(img8, [hull], -1, (255, 0, 0), 50)
edges = cv2.Canny(img8,100,200)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_list = []
for contour in contours:
    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
    area = cv2.contourArea(contour)
    if ((len(approx) > 8) & (len(approx) < 23) & (area > 30) ):
        contour_list.append(contour)
img6=cv2.imread(a)
img6=0*img6
cv2.drawContours(img6, contour_list,  -1, (255,0,0), 2)

#finding the circle in the image and its center and radius 
#creating ROI for image processing(we are just talking about red light and since the red light lies in the upper part hence the roi parameters will be taken accordingly)

centre=[]
radius=[]
points=[] #to form ROI for DL processing
image=cv2.imread(a)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)  
gray_blurred = cv2.blur(gray, (3, 3))
detected_circles = cv2.HoughCircles(gray_blurred,  cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 40, minRadius = 0, maxRadius = 0) 
if detected_circles is not None: 
    detected_circles = np.uint16(np.around(detected_circles)) 
    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
        centre.append([a,b])
        radius.append(r)
        points.append([[a-2*r,b-2*r],[a+2*r,b+4*r]])
        cv2.rectangle(image, (a-2*r,b-2*r), (a+2*r,b+4*r), (255,0,0), 5)
        #cv2.circle(image, (a, b), r, (0, 255, 0), 2) 
        #cv2.circle(image, (a, b), 1, (0, 0, 255), 3) #to show the centre


#creating ROI for image processing(we are just talking about red light and since the red light lies in the upper part hence the roi parameters will be taken accordingly)


    

print(centre,radius,points)
plt.imshow(image)
plt.show()
