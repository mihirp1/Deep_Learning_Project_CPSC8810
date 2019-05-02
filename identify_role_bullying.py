# Program   : Bounding Box creator for actors in Cyber Bullying
# Author    : Mihir Phatak & Netra Inamdar

import cv2
import os

LOCATION = "INPUT"
picture_ar = [LOCATION+"/"+s for s in os.listdir(LOCATION)]
O_LOCATION = "OUTPUT"

WRITING_STYLE = cv2.FONT_HERSHEY_SIMPLEX

aggressor_cas_file = "aggressor.xml"
victim_cas_file = "victim.xml"

# Loading the classifier of faces of aggressor and victim
aggressor_cas = cv2.CascadeClassifier(aggressor_cas_file)
victim_cas = cv2.CascadeClassifier(victim_cas_file)

for picture_file in picture_ar:
    picture_file_only_name = picture_file.split("/")[1]
    picture = cv2.imread(picture_file)

    WIGHT, DIMENSION, channel = picture.shape
    new_WIGHT = 500
    aspect_ratio = (new_WIGHT*1.0) / DIMENSION
    new_DIMENSION = int(WIGHT*aspect_ratio)
    resize = (new_WIGHT,new_DIMENSION)
    picture = cv2.resize(picture, resize, interpolation = cv2.INTER_AREA)

    picture_gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

    # detection of victim face
    victim_face_ar = victim_cas.detectMultiScale(
        picture_gray,
        scaleFactor = 1.15,
        minNeighbors = 4,
        minSize = (50,50)
        )

    # detection of aggressor face
    aggressor_face_ar = aggressor_cas.detectMultiScale(
        picture_gray,
        scaleFactor = 1.15,
        minNeighbors = 7,
        minSize = (80,80)
        )

    # marking victim faces
    for(XCORD,YCORD,WIDTH,HEIGHT) in victim_face_ar:
        cv2.rectangle(picture,(XCORD,YCORD),(XCORD+WIDTH,YCORD+HEIGHT),(0,255,0),2)
        cv2.putText(picture,"Victim",(XCORD,YCORD-10),WRITING_STYLE,0.55,(0,255,0),1)

    # marking aggressor faces
    for(XCORD,YCORD,WIDTH,HEIGHT) in aggressor_face_ar:
        cv2.rectangle(picture,(XCORD,YCORD),(XCORD+WIDTH,YCORD+HEIGHT),(0,0,255),2)
        cv2.putText(picture,"Aggressor",(XCORD,YCORD-10),WRITING_STYLE,0.55,(0,0,255),1)

    output_file = O_LOCATION+"/out_"+picture_file_only_name

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Images have been rendered")
