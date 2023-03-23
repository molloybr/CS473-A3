import cv2
import numpy as np


# Load the two images you want to stitch together
img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)


# Create SURF object
sift = cv2.SIFT_create()

# Find keypoints and descriptors in the two images using SURF
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

def extractTransform(im1_points, im2_points, im1_desc, im2_desc): 
    # Create a Brute Force Matcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match the descriptors using the Brute Force Matcher object
    matches = bf.match(des1, des2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x:x.distance)

    # Find the homography matrix using the homogenous DLT estimation method with singular value decomposition
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,2)

    A = np.zeros((len(matches) * 2, 9))

    for i in range(len(matches)):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A[1*2] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[1*2+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]


    A = np.asarray(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1,:].reshape(3, 3)
    
    return H


H_matrix = extractTransform(kp1, kp2, des1, des2)
im1_matrix = np.array([[1373, 1204],[1841, 1102],[1733,1213]])
print(H_matrix @ im1_matrix)

# Use the homography matrix to transform the second image and create the panorama
# result = cv2.warpPerspective(img2, H_matrix, (img1.shape[1] + img2.shape[1], img1.shape[0] + img2.shape[1]))
# result[0:img1.shape[0], 0:img1.shape[1]] = img1


