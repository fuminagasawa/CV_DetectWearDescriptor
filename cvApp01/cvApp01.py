#coding:utf-8

import numpy as np
import cv2
from sklearn.cluster import KMeans
import glob



# 画像から顔と体の領域を取得
def GetBodyRects( img):

    faceCascade = cv2.CascadeClassifier('./hori/haarcascade_frontalface_alt.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 3)

    faceRect = []
    bodyRect = []

    if len(faces) > 0:
        for rect in faces:

            faceRect.append( rect)

            faceCenterX = int((rect[0]+rect[0]+rect[2])/2 )
            curBody = [faceCenterX-rect[2], rect[1]+rect[3], faceCenterX+rect[2], rect[1]+rect[3]*4]

            bodyRect.append( curBody)
                       
    else:
        print("no face")
        return [], []

    return faceRect, bodyRect

# 領域を指定して画像を切り取る
def GetBodyImage( wholeImg, rect):

    bodyImg = wholeImg [rect[1]:rect[3],rect[0]:rect[2],:].copy()

    return bodyImg

# 画像をsiftにかけてKeyPointsとDescriptorsを取得
def GetSiftKeysOfBody( img, curBody):

    imgSiftInput = img [curBody[1]:curBody[3],curBody[0]:curBody[2],:].copy()

    imgSiftOutput= imgSiftInput.copy()

    imgSiftGray  = cv2.cvtColor(imgSiftInput,cv2.COLOR_BGR2GRAY)

 
    sift    = cv2.xfeatures2d.SIFT_create()
    kp,des  = sift.detectAndCompute( imgSiftGray,None)

    return kp, des

# 画像パスから切り取り済画像とKeyPointsとDescriptorsを取得
def GetSiftDescriptorsFromImage( path):

    wholeImg = cv2.imread( path, cv2.IMREAD_COLOR )

    faceRect, bodyRects = GetBodyRects( wholeImg)

    if(len(faceRect) == 0):
        return False, wholeImg, wholeImg, np.empty(1), np.empty(1)

    kp, des = GetSiftKeysOfBody( wholeImg, bodyRects[0])

 
    imgSiftInput = GetBodyImage( wholeImg, bodyRects[0])
    imgSiftOutput= imgSiftInput.copy()

    cv2.drawKeypoints(imgSiftInput ,kp, imgSiftOutput)

    
    return True, wholeImg, imgSiftInput, kp, des




if __name__ == '__main__':

 
    


    Descriptors = []

    """
    wholeImg, imgSiftInput, kp, des = GetSiftDescriptorsFromImage( "./hori/zozo_master/tops/knit-sweater/1.jpg")
    imgSiftOutput = imgSiftInput.copy()
    cv2.drawKeypoints(imgSiftInput ,kp, imgSiftOutput)
    
    Descriptors.append(des)

    cv2.imshow("Body-From", imgSiftInput)
    cv2.imshow("Body-Sift", imgSiftOutput)
    """
 
    

            
    # パス内の全ての"指定パス+ファイル名"と"指定パス+ディレクトリ名"を要素とするリストを返す
    files = glob.glob("./hori/zozo_master/tops/knit-sweater/*.jpg") # ワイルドカードが使用可能

    cnt = 0
    for file in files:
        print("processing : "+file)

        

        bodyDetected , wholeImg, imgSiftInput, kp, des = GetSiftDescriptorsFromImage( file)

        if( bodyDetected == True):
            imgSiftOutput = imgSiftInput.copy()
            cv2.drawKeypoints(imgSiftInput ,kp, imgSiftOutput)
            Descriptors.append( des)


    
    
    
    print("Detected Images" + str(len(Descriptors)))
    

#   print(Descriptors)
    
    
#    kmeans = KMeans(n_clusters=20, random_state=0).fit(des)
    
#    print(kmeans.cluster_centers_)
#    print(kmeans.labels_)
    
    
    
    
    
    
    
#    cv2.waitKey(0)
#    cv2.destroyAllWindows() 



