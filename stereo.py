import numpy as np
#==============No additional imports allowed ================================#

def get_ncc_descriptors(img, patchsize):
    '''
    Prepare normalized patch vectors for normalized cross
    correlation.

    Input:
        img -- height x width x channels image of type float32
        patchsize -- integer width and height of NCC patch region.
    Output:
        normalized -- height* width *(channels * patchsize**2) array

    For every pixel (i,j) in the image, your code should:
    (1) take a patchsize x patchsize window around the pixel,
    (2) compute and subtract the mean for every channel
    (3) flatten it into a single vector
    (4) normalize the vector by dividing by its L2 norm
    (5) store it in the (i,j)th location in the output

    If the window extends past the image boundary, zero out the descriptor
    
    If the norm of the vector is <1e-6 before normalizing, zero out the vector.

    '''
    h, w, c = img.shape
    offset = patchsize//2
    matrix = np.zeros((h, w, c*patchsize*patchsize))
    for i in range(offset, h-offset):
        for j in range(offset, w-offset):
            patch = img[i - offset:i+offset + 1, j - offset:j + offset + 1]
            means = np.mean(patch, axis=(0,1))
            patch = patch - means
            patch = patch.flatten()
            norm = np.linalg.norm(patch)
            matrix[i,j] = patch / (norm + 1e-8)
    return matrix


def compute_ncc_vol(img_right, img_left, patchsize, dmax):
    '''
    Compute the NCC-based cost volume
    Input:
        img_right: the right image, H x W x C
        img_left: the left image, H x W x C
        patchsize: the patchsize for NCC, integer
        dmax: maximum disparity
    Output:
        ncc_vol: A dmax x H x W tensor of scores.

    ncc_vol(d,i,j) should give a score for the (i,j)th pixel for disparity d. 
    This score should be obtained by computing the similarity (dot product)
    between the patch centered at (i,j) in the right image and the patch centered
    at (i, j+d) in the left image.

    Your code should call get_ncc_descriptors to compute the descriptors once.

    '''
    h, w, c = img_right.shape
    out = np.zeros((dmax, h, w))
    right = get_ncc_descriptors(img_right, patchsize)
    left = get_ncc_descriptors(img_left, patchsize)
    for d in range(dmax):
        for i in range (h):
            for j in range(w-d):
                out[d, i, j] = np.dot(right[i, j], left[i, j + d] )
                
    return out
    pass

def get_disparity(ncc_vol):
    '''
    Get disparity from the NCC-based cost volume
    Input: 
        ncc_vol: A dmax X H X W tensor of scores
    Output:
        disparity: A H x W array that gives the disparity for each pixel. 

    the chosen disparity for each pixel should be the one with the largest score for that pixel
    '''
    d, h, w = ncc_vol.shape
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            out[i, j] = np.argmax(ncc_vol[:, i, j], axis=0)
    return out
    pass