import cv2
import numpy as np
import opticalflow
import pickle


# 1. video to image
# 2.

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


# need for loop to open every images
for i in range(index):
    # im1 = cv2.imread("/home/moriyama/PycharmProjects/op_oc/img/picture001.jpg")
    # im2 = cv2.imread("/home/moriyama/PycharmProjects/op_oc/img/picture002.jpg")
    im1 = cv2.imread("/home/moriyama/PycharmProjects/op_oc/img/picture%03d" + format(i) + ".jpg")
    im2 = cv2.imread("/home/moriyama/PycharmProjects/op_oc/img/picture%03d" + format(i + 1) + ".jpg")

    im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    im1 = np.float64(im1 / 255)
    im2 = np.float64(im2 / 255)

# with open("/home/moriyama/PycharmProjects/op_oc/opticalflow/003.flo", "rb") as f:
#     flow = pickle.load(f)
flow = cv2.calcOpticalFlowFarneback(im1,im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

hsv = draw_hsv(flow)
im2w = warp_flow(im1*255, flow)
cv2.imwrite("./flow.jpg",hsv)
cv2.imwrite("./im1.jpg", im1*255)
cv2.imwrite("./im2.jpg", im2*255)
cv2.imwrite("./im2w.jpg", im2w)