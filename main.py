'''
Steps for alignment of 2 images
1. Import 2 images
2. Convert to greyscale
3. Initiate ORB Detector
4. Find keypoints and describe them
5. Match keypoints (brute force matcher) and sort them
6. RANSAC - reject bad keypoints
7. Register 2 images (use homology)
'''

# ORB

import cv2
import numpy as np

im1 = cv2.imread('input_images/DJI_20250530123037_0003_T.JPG')
im2 = cv2.imread('input_images/DJI_20250530123037_0003_Z.JPG')

img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Initiate ORB

orb = cv2.ORB_create(nfeatures=2000)

kp1 , des1 = orb.detectAndCompute(img1, None)
kp2 , des2 = orb.detectAndCompute(img2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match Descriptors

matches = matcher.match (des1, des2, None)

matches = sorted (matches , key = lambda x:x.distance) 

points1 = np.zeros ((len (matches), 2), dtype = np.float32)
points2 = np.zeros ((len (matches), 2), dtype = np.float32)

for i, match in enumerate (matches):
    points1 [i, :] = kp1 [match.queryIdx].pt
    points2 [i, :] = kp2 [match.trainIdx].pt

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

height , width , channels = im2.shape

im1Reg = cv2.warpPerspective (im1 , h, (width, height))

overlay = cv2.addWeighted(im2, 0.6, im1Reg, 0.4, 0)

cv2.imwrite('output_images/DJI_20250530123037_0003_AT.JPG', overlay)














