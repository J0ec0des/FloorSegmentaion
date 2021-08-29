import cv2
import numpy

#capture = cv.VideoCapture(0)
#for video input


gray = cv2.imread('Assets/IMG_3095.png', cv2.IMREAD_GRAYSCALE)

# other ways for grayscale img:
# gray = numpy.zeros(img.shape[:-1], dtype=img.dtype)
# for i, img in enumerate(img):
#     gray[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

graysmall = cv2.resize(gray, (800, 800))

blur = cv2.GaussianBlur(graysmall, (3,3),0)



print(gray.shape)
print(graysmall.shape)
print(blur.shape)


Eigen_dst = cv2.cornerEigenValsAndVecs(blur, 32, 3);

print(Eigen_dst.shape)
h, w = Eigen_dst.shape[:2]
print(Eigen_dst)
Eigen = Eigen_dst.reshape(h, w)
print(Eigen.shape)
print(Eigen)
cv2.imshow('1', Eigen)

# Canny:
Canny= cv2.Canny(blur, 100, 200)
cv2.imshow('canny', Canny)


cv2.imshow('blur', blur)

# cv2.imshow('EigenDetect', EigenDetect)

cv2.waitKey(0)
cv2.destroyAllWindows()

