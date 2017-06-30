#-*- coding:utf-8-*-
import sys

import cv2
import numpy as np
#添加注释用于测试

def preprocess(gray):
    # 1. Sobel���ӣ�x�������ݶ�
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    # 2. ��ֵ��
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    # 3. ���ͺ͸�ʴ�����ĺ˺���
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6))
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))

    # 4. ����һ�Σ�������ͻ��
    dilation = cv2.dilate(binary, element2, iterations = 1)

    # 5. ��ʴһ�Σ�ȥ��ϸ�ڣ������ߵȡ�ע������ȥ�������ֱ����
    erosion = cv2.erode(dilation, element1, iterations = 1)

    # 6. �ٴ����ͣ�����������һЩ
    dilation2 = cv2.dilate(erosion, element2, iterations = 3)

    # 6.5 �ٸ�ʴһ�Σ�ȥ�����ͨ��
    erosion2 = cv2.erode(dilation2,element3,iterations = 3)

    # 7. �洢�м�ͼƬ 
    cv2.imwrite("binary.png", binary)
 
    cv2.imwrite("dilation.png" , dilation)
    cv2.imwrite("erosion.png", erosion)
    cv2.imwrite("dilation2.png", dilation2)
    cv2.imwrite("erosion2.png",erosion2)

    return erosion2


def findTextRegion(img):
    region = []

    # 1. ��������
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. ɸѡ��Щ���С��
    for i in range(len(contours)):
        cnt = contours[i]
        # ��������������
        area = cv2.contourArea(cnt) 

        # ���С�Ķ�ɸѡ��
        if(area < 1000):
            continue

        # �������ƣ����ú�С
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # �ҵ���С�ľ��Σ��þ��ο����з���
        rect = cv2.minAreaRect(cnt)
        print "rect is: "
        print rect

        # box���ĸ��������
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)

        # ����ߺͿ�
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # ɸѡ��Щ̫ϸ�ľ��Σ����±��
        if(height > width * 1.2):
            continue

        region.append(box)

    return region


def detect(img):
    # 1.  ת���ɻҶ�ͼ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. ��̬ѧ�任��Ԥ������õ����Բ��Ҿ��ε�ͼƬ
    dilation = preprocess(gray)

    # 3. ���Һ�ɸѡ��������
    region = findTextRegion(dilation)
    num_region = len(region)

    # 4. �����߻�����Щ�ҵ�������,������ǵ�ƽ����
    num = 0
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        x1 = box[0][0]
        y1 = box[1][1]
        y2 = box[0][1]
        x2 = box[3][0] 
        roi = gray[y1:y2,x1:x2]
        num = num + num_corner(roi)
    average_corner = num/num_region
    

    # 5. ���ݽǵ���Ŀ��ɸѡ�����п��ܵ���������
    i = 1
    for box in region:
        x1 = box[0][0]
        y1 = box[1][1]
        y2 = box[0][1]
        x2 = box[3][0] 
        roi = gray[y1:y2,x1:x2]
        if num_corner(roi)>= (average_corner*0.4):
            cv2.imwrite(str(i)+'.png', roi)
        i = i + 1

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)

    # ��������ͼƬ
    cv2.imwrite("contours.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def num_corner(gray):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return len(dst)


if __name__ == '__main__':
    # ��ȡ�ļ�
    #imagePath = sys.argv[1]
    img = cv2.imread('D:/text7.jpg')
    detect(img)