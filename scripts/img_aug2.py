import cv2
from matplotlib import pyplot

img = cv2.imread('D:\\capstone project\\Flaskex-master\\Flaskex-master\\dataset\\test\\c5_1.jpg')
print(type(img))
# <class 'numpy.ndarray'>
pyplot.subplot(330 + 1 + 0)
print(img.shape)
# (225, 400, 3)
pyplot.imshow(img)

pyplot.subplot(330 + 1 + 3)
img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite('../dataset/train/lena_cv_rotate_90_clockwise.jpg', img_rotate_90_clockwise)
pyplot.imshow(img_rotate_90_clockwise)
# True

pyplot.subplot(330 + 1 + 4)
img_rotate_90_counterclockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite('../dataset/train/lena_cv_rotate_90_counterclockwise.jpg', img_rotate_90_counterclockwise)
pyplot.imshow(img_rotate_90_counterclockwise)
# True
pyplot.subplot(330 + 1 + 5)
img_rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
cv2.imwrite('../dataset/train/lena_cv_rotate_180.jpg', img_rotate_180)
pyplot.imshow(img_rotate_180)
# True
pyplot.show()