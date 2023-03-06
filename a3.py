import cv2
import numpy as np

# Load the two images you want to stitch together
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

# Create SURF object
surf = cv2.xfeatures2d.SURF_create()

# Find keypoints and descriptors in the two images using SURF
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

# Create a Brute Force Matcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors using the Brute Force Matcher object
matches = bf.match(des1, des2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x:x.distance)

# Find the homography matrix using the homogenous least squares transformation estimation method with singular value decomposition
src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
A = np.zeros((len(matches) * 2, 9))
for i in range(len(matches)):
    x, y = src_pts[i][0]
    u, v = dst_pts[i][0]
    A[2 * i] = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
    A[2 * i + 1] = [0, 0, 0, x, y, 1, -v*x, -v*y, -v]
U, s, Vh = np.linalg.svd(A)
L = Vh[-1,:] / Vh[-1,-1]
H = L.reshape(3,3)

# Use the homography matrix to transform the second image and create the panorama
result = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
result[0:img1.shape[0], 0:img1.shape[1]] = img1

# Display the resulting panorama
cv2.imshow("Panorama", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
