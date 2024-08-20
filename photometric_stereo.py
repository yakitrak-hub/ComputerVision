import numpy as np
##======================== No additional imports allowed ====================================##






def photometric_stereo_singlechannel(I, L):
    #L is 3 x k
    #I is k x n
    G = np.linalg.inv(L @ L.T) @ L @ I
    # G is  3 x n 
    albedo = np.sqrt(np.sum(G*G, axis=0))

    normals = G/(albedo.reshape((1,-1)) + (albedo==0).astype(float).reshape((1,-1)))
    return albedo, normals


def photometric_stereo(images, lights):
    '''
        Use photometric stereo to compute albedos and normals
        Input:
            images: A list of N images, each a numpy float array of size H x W x 3
            lights: 3 x N array of lighting directions. 
        Output:
            albedo, normals
            albedo: H x W x 3 array of albedo for each pixel
            normals: H x W x 3 array of normal vectors for each pixel

        Assume light intensity is 1.
        Compute the albedo and normals for red, green and blue channels separately.
        The normals should be approximately the same for all channels, so average the three sets
        and renormalize so that they are unit norm

    '''


    height, width, channels = images[0].shape
    N= len(images)


    red = np.concatenate([image[:,:,[0]].reshape(1,-1) for image in images], axis=0) 
    blue = np.concatenate([image[:,:,[1]].reshape(1,-1) for image in images], axis=0) 
    green = np.concatenate([image[:,:,[2]].reshape(1,-1) for image in images], axis=0) 

            

    r_albedos, r_normals = photometric_stereo_singlechannel(red,lights)
    b_albedos, b_normals = photometric_stereo_singlechannel(blue,lights)
    g_albedos, g_normals = photometric_stereo_singlechannel(green,lights)


    reshaped_ralb = r_albedos.reshape(-1,1)
    reshaped_balb=b_albedos.reshape(-1,1)
    reshaped_galb = g_albedos.reshape(-1,1)

    albedos = np.concatenate([reshaped_ralb,reshaped_balb,reshaped_galb],axis=1)
    albedos = albedos.reshape(images[0].shape)
    avg_normal = (r_normals + b_normals+ g_normals)/3
    norm_normal = np.sqrt(np.sum(avg_normal*avg_normal,keepdims=True,axis=0))
    norm_divby0 = norm_normal +(norm_normal==0).astype(float)
    normals = avg_normal/norm_divby0
    normals = normals.T.reshape(images[0].shape)
    return albedos, normals




    



