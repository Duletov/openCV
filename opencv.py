import numpy as np
import cv2
import random

img = cv2.imread('picture.jpg')
img2 = img.copy()

rows, cols, channels = img.shape

#найти все orb features точки на изображении
orb = cv2.ORB_create()
kp = orb.detect(img,None)
kp, des = orb.compute(img, kp)
orbP = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv2.imwrite('orb.jpg', orbP)

#найти все sift features точки на изображении
sift = cv2.SIFT_create()
gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
kp = sift.detect(gray,None)
siftP=cv2.drawKeypoints(gray,kp,img2)
cv2.imwrite('sift.jpg', siftP)

# найти canny edges на изображенни
edges = cv2.Canny(img, 100, 200)
cv2.imwrite('edges.jpg', edges)

#перевести в grayscale
cv2.imwrite('gray.jpg', gray)

#перевести изорбражение в hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('hsv.jpg', hsv)

#отразить изображение по правой границе
#отразить изображение по нижней границе
horiz = img[:,::-1]
vert = cv2.flip(img,0)
cv2.imwrite('horiz.jpg', horiz)
cv2.imwrite('vert.jpg', vert)

#повернуть изображение на 45 градусов
M = cv2.getRotationMatrix2D((rows / 2, cols / 2), 45, 1.0)
rotate = cv2.warpAffine(img, M, (rows, cols))
cv2.imwrite('rotate.jpg', rotate)

# повернуть изображение на 30 градусов вокруг заданной точки
M = cv2.getRotationMatrix2D((rows / 3, cols / 3), 30, 1.0)
rotate2 = cv2.warpAffine(img, M, (rows, cols))
cv2.imwrite('rotate2.jpg', rotate2)

#сместить изображение но 10 пикселей вправо
M = np.float32([[1, 0, 10], [0, 1, 0]])
move = cv2.warpAffine(img, M, (rows, cols))
cv2.imwrite('move.jpg', move)

#изменить яркость изоббражения
h, s, v = cv2.split(hsv)
v[v > 205] = 255
v[v <= 205] += 50
hsv2 = cv2.merge((h, s, v))
brightness = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
cv2.imwrite("brightness.jpg", brightness)

#изменить контрасть изображения
contrast = cv2.addWeighted(img, 1.25, img, 0, 0)
cv2.imwrite("contrast.jpg", contrast)

#сделать гамма-перобразование 
M = [((i / 255) ** (1/4.0)) * 255 for i in range(256)]
M = np.array(M, np.uint8)
gamma = cv2.LUT(img, M)
cv2.imwrite("gamma.jpg", gamma)

#сделать гистограмную эквайлизацию
histequ = cv2.equalizeHist(gray)
cv2.imwrite("histequ.jpg", histequ)

#изменить баланс белого, сделать более "теплую" картинку
originalValues = np.array([0, 50, 100, 150, 200, 255])
maxValues = np.array([0, 80, 150, 190, 220, 255])
minValues = np.array([0, 20, 40, 75, 150, 255])
allValues = np.arange(0, 256)
redLookupTable = np.interp(allValues, originalValues, maxValues)
blueLookupTable = np.interp(allValues, originalValues, minValues)
B, G, R = cv2.split(img)
R = np.uint8(cv2.LUT(R, redLookupTable))
B = np.uint8(cv2.LUT(B, blueLookupTable))
warm = cv2.merge([B, G, R])
cv2.imwrite("warm.jpg", warm)

#изменить баланс белого, сделать более "холодную" картинку
redLookupTable = np.interp(allValues, originalValues, minValues)
blueLookupTable = np.interp(allValues, originalValues, maxValues)
B, G, R = cv2.split(img)
R = np.uint8(cv2.LUT(R, redLookupTable))
B = np.uint8(cv2.LUT(B, blueLookupTable))
cold = cv2.merge([B, G, R])
cv2.imwrite("cold.jpg", cold)

#изменить цветовую палитру по заданному шаблону
LH = cv2.getTrackbarPos("l_h", "Tracking")
LS = cv2.getTrackbarPos("l_s", "Tracking")
LV = cv2.getTrackbarPos("l_v", "Tracking")
UH = cv2.getTrackbarPos("u_h", "Tracking")
US = cv2.getTrackbarPos("u_s", "Tracking")
UV = cv2.getTrackbarPos("u_v", "Tracking")
lower_border = np.array([LH, LS, LV])
upper_border = np.array([UH, US, UV])
mask = cv2.inRange(hsv, lower_border, upper_border)
palette = cv2.bitwise_and(img, img, mask)
cv2.imwrite("palette.jpg", palette)

#сделать бинаризацию изображения
th, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite("thresh.jpg", thresh)

#найти контуры на бинаризированном изображении
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
img1 = img.copy()
binCountours = cv2.drawContours(img1, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imwrite("binCountours.jpg", binCountours)

#нати контуры на изображении, применив фильтры (Лапласиан)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
cv2.imwrite("laplacian.jpg", laplacian)

#сделать размытие изображения
kernel = np.ones((5, 5), np.float32) / 25
blurr = cv2.blur(img, (5, 5))
cv2.imwrite("blur.jpg", blurr)

#сделать фильтрацию изображения при помощи Фурье преобразоваия, оставить только быстрые частоты
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
mask = np.zeros((rows,cols),np.uint8)
mask[(rows//2)-(rows//4):(rows//2)+(rows//4), (cols//2)-(cols//4):(cols//2)+(cols//4)+1] = 1
fshift = fshift*mask
f_ishift = np.fft.ifftshift(fshift)
LPF = np.fft.ifft2(f_ishift)
LPF = np.real(LPF)
cv2.imwrite("LPF.jpg", LPF)

#сделать фильтрацию изображения при помощи Фурье преобразоваия, оставить только медленные частоты
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
fshift[(rows//2)-(rows//4):(rows//2)+(rows//4)+1, (cols//2)-(cols//4):(cols//2)+(cols//4)+1] = 0
f_ishift = np.fft.ifftshift(fshift)
HPF = np.fft.ifft2(f_ishift)
HPF = np.real(HPF)
cv2.imwrite("HPF.jpg", HPF)

#применить операцию эрозии к изображению
kernel = np.ones((5, 5), 'uint8')
erod = cv2.erode(img, kernel, iterations=1)
cv2.imwrite("erode.jpg", erod)

#применить операцию диляции к изображению
kernel = np.ones((5, 5), 'uint8')
dilat = cv2.dilate(img, kernel, iterations=1)
cv2.imwrite("dilate.jpg", dilat)


#cv2.imshow('image', LPF)
#k = cv2.waitKey(0)
