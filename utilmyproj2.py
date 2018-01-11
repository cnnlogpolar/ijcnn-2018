import cv2
import numpy as np

def aplicalogpolar(img,radius=40, x=0, y=0):


     
    #src2 = cv2.logPolar(img (x, y), radius, cv2.WARP_FILL_OUTLIERS)
    dst = cv2.logPolar(img, (x, y), radius, cv2.WARP_FILL_OUTLIERS)
    src2 = cv2.logPolar(dst, (x, y), radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    #print ('x:',x)
    #print ('y:',y)
    #print ('r:',radius)
    return src2



def aplicasift(img):


    print(img.dtype)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(gray.shape)

    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray, None)
    #print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
    
    pts = [p.pt for p in kps]
    
    #img=cv2.drawKeypoints(gray,kps,np.array([]), (0,0,255))

    pontos=np.median(pts, axis=0)
    radius=pontos[0]/3 
    x=pontos[0] 
    y=pontos[1]
    return radius, x,y


def aplicaCAMshift(img):
 
    lines=img.shape[0]
    cols=img.shape[1]
    #channels=img.shape[2]
    M00 = 0.0
    M10 = 0.0
    M01 = 0.0
    Center_x=[]
    Center_y=[]
    
    #for c in range(0,channels):
    i = range(lines)
    i = np.reshape(i,(lines,1))
    i = np.tile(i,(1,cols))
    M00 = np.sum(img)
    M10 = np.sum(np.multiply(i, img))
    j = range(cols)
    j = np.tile(j,(lines,1))
    M01 = np.sum(np.multiply(j, img))
#    for i in range(lines):  
#        for j in range(cols):  
#            M00 += img[i, j]
#            M10 += i * img[i, j]
#            M01 += j * img[i, j]
    Center_x.append(M01 / M00)
    Center_y.append(M10 / M00)
    #    M00 = 0.0
    #    M10 = 0.0
    #    M01 = 0.0


def logpolar_naive(image, i_0, j_0, p_n=None, t_n=None):
    (i_n, j_n) = image.shape[:2]
    
    i_c = max(i_0, i_n - i_0)
    j_c = max(j_0, j_n - j_0)
    d_c = (i_c ** 2 + j_c ** 2) ** 0.5

    #if p_n == None:
        #p_n = int(ceil(d_c))
    
    #if t_n == None:
        #t_n = j_n
    p_n = 50
    t_n = 50
    p_s = np.log(d_c) / p_n
    t_s = 2.0 * np.pi / t_n
    
    transformed = np.zeros((p_n, t_n) + image.shape[2:], dtype=image.dtype)

    p = range(0, p_n)
    p = np.reshape(p, (p_n, 1))
    p = np.tile(p, (1, t_n))
    p_exp = np.exp(p * p_s)
    t = range(0, t_n)
    t = np.tile(t, (p_n, 1))
    t_rad = t * t_s
    i = i_0 + np.multiply(p_exp, np.sin(t_rad))
    j = j_0 + np.multiply(p_exp, np.cos(t_rad))
    cond = np.logical_and(np.logical_and(np.greater_equal(i,0),np.lesser(i,i_n)),np.logical_and(np.greater_equal(j,0),np.lesser(j,j_n)))
    transformed[p,t] = np.where(cond, image[i,j])
    

    for p in range(0, p_n):
        p_exp = np.exp(p * p_s)
        for t in range(0, t_n):
            t_rad = t * t_s

            i = int(i_0 + p_exp * np.sin(t_rad))
            j = int(j_0 + p_exp * np.cos(t_rad))

            if 0 <= i < i_n and 0 <= j < j_n:
                transformed[p, t] = image[i, j]

    return transformed


def logpolar_naive_inv(image, i_0, j_0, i_n, j_n):
    (p_n, t_n) = image.shape[:2]
    
    i_c = max(i_0, i_n - i_0)
    j_c = max(j_0, j_n - j_0)
    d_c = (i_c ** 2 + j_c ** 2) ** 0.5

    p_s = np.log(d_c) / p_n
    t_s = 2.0 * np.pi / t_n
    
    transformed = np.zeros((i_n, j_n) + image.shape[2:], dtype=image.dtype)

    for i in range(0, i_n):
        p_exp1 = (i - i_0)
        for j in range(0, j_n):
            p_exp2 = (j - j_0)
            t_rad = np.arctan2(p_exp1, p_exp2)
            t = int(t_rad / t_s)
            if t < 0:
                t = t_n + t
            if np.sin(t_rad) == 0:
                if np.cos(t_rad) == 0:
                    p_exp = 0
                else:
                    p_exp == p_exp2 / np.cos(t_rad)
            else:
                p_exp = p_exp1 / np.sin(t_rad)
            if p_exp == 0:
                p = p_n+1
            else:
                p = int(np.log(p_exp) / p_s)
            if 0 <= p < p_n and 0 <= t < t_n:
                transformed[i,j] = image[p,t]

    return transformed
