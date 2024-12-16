import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder


# cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)

# Load the captured image
cap = cv2.imread("coin.jpg")  # Replace with the actual path to your image
img = cv2.resize(cap, (500, 400))  # Resize to match the desired dimensions

myColorFinder = ColorFinder(trackBar=False)
hsvVals = {'hmin': 8, 'smin': 10, 'vmin': 50, 'hmax': 30, 'smax': 255, 'vmax': 226}
# hsvVals = {'hmin': 10, 'smin': 55, 'vmin': 215, 'hmax': 42, 'smax': 255, 'vmax': 255}


def empty():
    pass

cv2.namedWindow( "Settings")
cv2.resizeWindow('Settings',240,240)
cv2.createTrackbar ("Threshold1",'Settings',219,255,empty)
cv2.createTrackbar ("Threshold2",'Settings',100,255,empty)

def preProcessingImg(img):
    processed_img = cv2.GaussianBlur(img,(5,5),3)
    thresh1= cv2.getTrackbarPos("Threshold1",'Settings')
    thresh2= cv2.getTrackbarPos("Threshold2",'Settings')
    processed_img = cv2.Canny(processed_img,threshold1=thresh1,threshold2=thresh2)

    kernel = np.ones((2,3),np.uint8)
    processed_img = cv2.dilate(processed_img,kernel=kernel, iterations=1)
    processed_img = cv2.morphologyEx(processed_img,cv2.MORPH_CLOSE,kernel)
    return processed_img

while True:
    # success , img = cap.read()
    processed_img = preProcessingImg(img)
    imgCountours, conFound  = cvzone.findContours(img, processed_img,minArea=20)
    
    imgCount = np.zeros((480,640,3),np.uint8)
    totalMoney = 0

    if conFound:
        for contour in conFound:
            peri = cv2.arcLength(contour['cnt'],True)
            approx = cv2.approxPolyDP(contour['cnt'],0.02*peri, True)

            if len(approx)>5:
                area = contour['area']
                
                imgColor , _ = myColorFinder.update(img, hsvVals)

                if area>2050:
                    totalMoney +=5

                elif 2050<area<2500:
                    totalMoney +=1
                else:
                    totalMoney +=2
    print("totalMoney:",totalMoney)
    cvzone.putTextRect(img=imgCount,text=f"Rs.{totalMoney}",pos=(100,250),scale=10,thickness=7,offset=30)

    img_stacked = cvzone.stackImages([img,processed_img,imgCountours,imgCount],2,1)
    # cvzone.putTextRect(img=img_stacked,text=f"Rs.{totalMoney}",pos=(100,50))
    cv2.imshow('Image',img_stacked)
    # cv2.imshow('Image color',imgColor)
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
