import os 
import cv2
import sys as s
def ConvertToGrey(path,out):
    for img in os.listdir(path):
        temp = cv2.imread(os.path.join(path,img))
        grey_image = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite(grey_image,os.path.join(out,img))

if __name__ == "__main__":
    ConvertToGrey(s.argv[1],s.argv[2])