import cv2
import numpy as np
import random
from a2 import transform_image
import tracemalloc

def estimateTransform(im1_points, im2_points):
    num_points = len(im1_points)
    P = np.zeros((2*num_points, 9))
    r = np.zeros((2*num_points,))
    for i in range(num_points):
        x, y = im1_points[i][0]
        u, v = im2_points[i][0]
        P[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        P[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
        r[2*i] = -u
        r[2*i+1] = -v
    _, _, vt = np.linalg.svd(P)
    q = vt.T[:, -1]
    A = np.reshape(q, (3, 3))
    return A / A[2, 2]

def estimateTransformRANSAC(im1_points, im2_points):
    num_points = len(im1_points)
    threshold = 5
    k = im1_points.shape[0]
    best_inliers = []
    best_A = None
    for i in range(1000):  # Number of RANSAC iterations
        # Select k random correspondences
        rand_indices = random.sample(range(num_points), k)
        rand_im1_pts = np.float32([im1_points[j] for j in rand_indices]).reshape(-1, 1, 2)
        rand_im2_pts = np.float32([im2_points[j] for j in rand_indices]).reshape(-1, 1, 2)

        # Estimate transformation using the k correspondences
        A = estimateTransform(rand_im1_pts, rand_im2_pts)

        # Compute error for all correspondences
        errors = np.zeros(num_points)
        for j in range(num_points):
            im1_pt = np.array([im1_points[j]])
            im2_pt = np.array([im2_points[j]])
            transformed_pt = cv2.perspectiveTransform(im1_pt.reshape(1, 1, 2), A)
            errors[j] = np.linalg.norm(transformed_pt - im2_pt)

        # Count inliers
        inliers = np.where(errors < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_A = estimateTransform(
                np.float32([im1_points[j] for j in inliers]).reshape(-1, 1, 2),
                np.float32([im2_points[j] for j in inliers]).reshape(-1, 1, 2)
            )

    return best_A

tracemalloc.start()
print(tracemalloc.get_traced_memory())

img1 = cv2.imread("Image1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("Image2.jpg", cv2.IMREAD_GRAYSCALE)

img1 = cv2.resize(img1, (0,0), fx=0.1, fy=0.1)
img2 = cv2.resize(img2, (0,0), fx=0.1, fy=0.1)

sift = cv2.xfeatures2d.SIFT_create()

print(tracemalloc.get_traced_memory())

kp1, feat1 = sift.detectAndCompute(img1, None)
kp1 = [k for k in kp1 if k.pt is not None]
kp2, feat2 = sift.detectAndCompute(img2, None)
kp2 = [k for k in kp2 if k.pt is not None]

bf = cv2.BFMatcher(crossCheck = True)
matches = bf.match(feat1, feat2)

good_matches = [m for m in matches if kp1[m.queryIdx].pt is not None and kp2[m.trainIdx].pt is not None]
good_matches = sorted(good_matches, key=lambda x: x.distance)

img1_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
img2_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

del kp1, kp2, feat1, feat2, matches, bf, good_matches, sift, img1, img2

print(tracemalloc.get_traced_memory())

A = estimateTransformRANSAC(img1_pts, img2_pts)

del img1_pts, img2_pts

print(tracemalloc.get_traced_memory())

A_inv = np.linalg.inv(A)

del A

print(tracemalloc.get_traced_memory())

img2 = cv2.imread("Image2.jpg", cv2.IMREAD_GRAYSCALE)

img2 = cv2.resize(img2, (0,0), fx=0.1, fy=0.1)

print(tracemalloc.get_traced_memory())

im2_transformed = transform_image(img2, A_inv, 'homography')

cv2.imshow('Image 2 Transformed', im2_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()

