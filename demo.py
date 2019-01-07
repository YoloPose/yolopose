from yolo_person import *
from inception_pose import *
import cv2


def apply_results( img, results):
    img_cp = img.copy()
    for i in range(len(results)):
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3]) // 2
        h = int(results[i][4]) // 2

        cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
        cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img_cp

def draw_line(x,y,w,h,pred,img,p1,p2):
    cv2.line(img, (int(x + w * pred[0][2*p1]), int(y + h * pred[0][2*p1+1])),(int(x + w * pred[0][2*p2]), int(y + h * pred[0][2*p2+1])),color=(0,255,0),thickness=3)


def draw_result(x,y,w,h,pred,img):
    # Draw joints
    for i in range(16):
        cv2.circle(img,(int(x+w*pred[0][2*i]),int(y+h*pred[0][2*i+1])),5,thickness=-1,color=(255,0,0))

    # Draw lines
    draw_line(x, y, w, h, pred, img, 0, 1)
    draw_line(x, y, w, h, pred, img, 1, 2)
    draw_line(x, y, w, h, pred, img, 2, 6)
    draw_line(x, y, w, h, pred, img, 6, 3)
    draw_line(x, y, w, h, pred, img, 3, 4)
    draw_line(x, y, w, h, pred, img, 4, 5)
    draw_line(x, y, w, h, pred, img, 7, 6)
    draw_line(x, y, w, h, pred, img, 7, 8)
    draw_line(x, y, w, h, pred, img, 10, 11)
    draw_line(x, y, w, h, pred, img, 11, 12)
    draw_line(x, y, w, h, pred, img, 12, 8)
    draw_line(x, y, w, h, pred, img, 8, 13)
    draw_line(x, y, w, h, pred, img, 13, 14)
    draw_line(x, y, w, h, pred, img, 14, 15)
    draw_line(x, y, w, h, pred, img, 8, 9)


if __name__ =='__main__':
    yolo = YOLO_TF(False, False)
    pose = Inception()
    print('*'*400)
    img = cv2.imread('demo/2.jpg')

    results = yolo.detect_from_cvmat(img)
    for result in results:
        x,y = result[1], result[2]
        w,h = result[3], result[4]
        x1 = int( x - w/2 )
        x2 = int( x + w/2 )
        y1 = int( y - h/2 )
        y2 = int( y + h/2 )

        if result[0] == 'person' and result[5] > 0.7 and result[5] < 0.8:
            print(x,y,w,h)
            print(x1,y1,x2,y2)
            img_part = cv2.imread('data/processed_images/005808361.jpg')
            print("img_part size: ",img_part.shape)
            pred = pose.detect(img_part)
            print(pred)
            draw_result(x1-170, y1-50, 710, h+20, pred, img)

    img = apply_results(img, results)
    print("apply")
    cv2.imshow("test", img)
    cv2.waitKey(0)
