import torch
import torch.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
# import flowlib
import pickle
import numpy as np
import cv2


'''
opticalflow

'''

video_capture = cv2.VideoCapture("/home/moriyama/Downloads/Boat1193.mp4")
output_file = "./output.mp4"



ret, frame1 = video_capture.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
index = 0

while(video_capture.isOpened()):
    index += 1
    ret, frame2 = video_capture.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    with open("./opticalflow/%03d.flo" %(index
                                         ), 'wb') as f:
        pickle.dump(flow,f)

    if ret == True:
        cv2.imwrite("./img/picture%03d"%(index)+".jpg", frame2)



    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

video_capture.release()
cv2.destroyAllWindows()



'''
occlusion mask
'''




