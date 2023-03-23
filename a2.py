import cv2
import numpy as np
import scipy

def transform_image(input_image, transform_matrix, transform_type) -> cv2.Mat:
    """
    Transform an image using a given transform matrix and type.

    :param input_image: The image to be transformed.
    :param transform_matrix: The transform matrix.
    :param transform_type: The type of transform to be applied.
    :return: The transformed image.
    """

 
    width = input_image.shape[1]
    height = input_image.shape[0]

    orig_coords = np.meshgrid(np.arange(width), np.arange(height))

    transform_types = ['translation', 'rotation', 'scaling', 'shear', 'affine', 'reflection', 'homography']

    assert transform_type in transform_types, 'Invalid transform type.'

    # Step 1: Create placeholder for output image

    new_width, new_height, shift_x, shift_y = get_i_prime(input_image, transform_matrix)
    shifted_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]])
    inverse_shifted_matrix = np.array([[1, 0, -shifted_matrix[0, 2]], [0, 1, -shifted_matrix[1, 2]], [0, 0, 1]])

    # Step 2/3: Select correct type of transform matrix, invert transform matrix
    if transform_type == 'translation':
        inverse_transform_matrix = np.array([[1, 0, -transform_matrix[0, 2]], [0, 1, -transform_matrix[1, 2]], [0, 0, 1]])
    elif transform_type == 'rotation':
        inverse_transform_matrix = np.array([[transform_matrix[0, 0], transform_matrix[1, 0], 0], [transform_matrix[0, 1], transform_matrix[1, 1], 0], [0, 0, 1]])
    elif transform_type == 'scaling':
        inverse_transform_matrix = np.array([[1/transform_matrix[0, 0], 0, 0], [0, 1/transform_matrix[1, 1], 0], [0, 0, 1]])
    elif transform_type == 'shear':
        inverse_transform_matrix = np.array([[1, -transform_matrix[0, 1], 0], [-transform_matrix[1, 0], 1, 0], [0, 0, 1]])
    elif transform_type == 'affine':
        inverse_transform_matrix = np.linalg.inv(transform_matrix)
    elif transform_type == 'reflection':
        inverse_transform_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif transform_type == 'homography':
        inverse_transform_matrix = np.linalg.inv(transform_matrix)

    # Step 4/5: For each (x', y') in I', find corresponding (x, y) in I, set I'(x', y') = I(x, y) using bilinear interpolation
    
    new_image = cv2.warpPerspective(input_image, inverse_transform_matrix.astype('float32') @ inverse_shifted_matrix, (new_width, new_height), flags=cv2.WARP_INVERSE_MAP)
    cv2.imshow('My Window', new_image)
    cv2.waitKey(0)

    return new_image

def get_i_prime(img, transform_matrix) -> cv2.Mat:
    img_corners = np.array([[0, 0, 1], [0, img.shape[0], 1], [img.shape[1], 0, 1], [img.shape[1], img.shape[0], 1]])

    new_corners = transform_matrix @ img_corners.T

    min_x = int(new_corners[0].min())
    max_x = int(new_corners[0].max())
    min_y = int(new_corners[1].min())
    max_y = int(new_corners[1].max())
    shift_x = 0
    shift_y = 0
    if min_x < 0:
        shift_x = abs(min_x)
        min_x += int(shift_x)
        max_x += int(shift_x)

    if min_y < 0:
        shift_y = abs(min_y)
        min_y += int(shift_y)
        max_y += int(shift_y)


    newH = max_y - min_y
    newW = max_x - min_x
    
    
    return newW, newH, shift_x, shift_y




if __name__ == '__main__':
    SIZE = (1920, 1080)

    #Resizing each image to be 1920 x 1080 for part 2.1
    img1_resized = cv2.resize(cv2.imread('Image1.png', cv2.IMREAD_GRAYSCALE), SIZE)
    img2_resized = cv2.resize(cv2.imread('Image2.png', cv2.IMREAD_GRAYSCALE), SIZE)
    img3_resized = cv2.resize(cv2.imread('Image3.jpeg', cv2.IMREAD_GRAYSCALE), SIZE)

    #Images of original size for the rest of part 2
    img1 = cv2.imread('Image1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('Image2.png', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('Image3.jpeg', cv2.IMREAD_GRAYSCALE)


    #All transformation matrices 
    reflect_y = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_30deg = np.array([[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]])
    shear = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    #These three transformations are the transformations that are composed for part 2.5
    composition_translation = np.array([[1, 0, 300], [0, 1, 500], [0, 0, 1]])
    composition_rotation = np.array([[0.940, 0.342, 0], [-0.342, 0.940, 0], [0, 0, 1]])
    composition_scaling = np.array([[0.5, 0, 0],[0, 0.5, 0], [0, 0, 1]])
    #Final composed transformation for part 2.5
    final_composition = composition_scaling @ composition_rotation @ composition_translation

    affine1 = np.array([[1, .4, .4], [.1, 1, .3], [0, 0, 1]])
    affine2 = np.array([[2.1, -.35, -.1], [-.3, .7, .3], [0, 0, 1]])
    homography1 = np.array([[.8, .2, .3], [-.1, .9, -.1], [0.0005, -0.0005, 1]])
    homography2 = np.array([[29.25, 13.95, 20.25], [4.95, 35.55, 9.45], [0.045, 0.09, 45]])

    #Function calls and images outputs for parts 2.2-2.7 for img1, simply replace img1 with img2 or img3
    reflected_image = transform_image(img3, reflect_y, 'reflection') 
    rotated_image = transform_image(img3, rot_30deg, 'rotation') 
    sheared_image = transform_image(img3, shear, 'shear')
    composed_image = transform_image(img3, final_composition, 'affine')
    affine_image_one = transform_image(img3, affine1, 'affine') 
    affine_image_two = transform_image(img3, affine2, 'affine')
    homography_image_one = transform_image(img3, homography1, 'homography') 
    homography_image_two = transform_image(img3, homography2, 'homography')

    #Writes image to directory, just replace first value with filename and second with the return from a transform_image above
    cv2.imwrite("image3Reflected.png", reflected_image)
    cv2.imwrite("image3Rotation.png", rotated_image)
    cv2.imwrite("image3Shear.png", sheared_image)
    cv2.imwrite("image3Composition.png", composed_image)
    cv2.imwrite("image3Affine1.png", affine_image_one)
    cv2.imwrite("image3Affine2.png", affine_image_two)
    cv2.imwrite("image3Homography1.png", homography_image_one)
    cv2.imwrite("image3Homography2.png", homography_image_two)

