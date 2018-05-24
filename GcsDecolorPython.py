from skimage.transform import resize
from scipy import ndimage
import math
import matplotlib.pyplot as plt
import scipy
import cv2
import numpy as np

# Example
#   ==========
#   gIm  = gcsdecolor2python(formato_imagen('Paint3x3.png'), 0.25)
#   Im = formato_imagen('Paint3x3.png')
#   plt.imshow(gIm)
#   plt.imshow(Im)


def formato_imagen_in(path):
    f = plt.imread(path)
    f = f.transpose()
    f[0] = f[0].transpose()
    f[1] = f[1].transpose()
    f[2] = f[2].transpose()
    return f


def formato_imagen_out(f):
    f[0] = f[0].transpose()
    f[1] = f[1].transpose()
    f[2] = f[2].transpose()
    f = f.transpose()
    return f



def matrix_to_vcolumna(matrix_n):
    matrix_n = np.matrix(matrix_n)
    vector_columna = np.ones((matrix_n.shape[0]*matrix_n.shape[1], 1))
    c = 0
    for j in range(matrix_n.shape[0]):
        for k in range(matrix_n.shape[1]):
            vector_columna[c, 0] = matrix_n[j, k]
            c += 1
    return vector_columna


def column(matrix_n, c):
    columna = []
    for j in range(matrix_n.shape[0]):
        columna.append(matrix_n[j, c])
    return columna


def gcsdecolor2python(im):
    return gcsdecolor2python(im, 0.25)


def gcsdecolor2python(im, Lpp_):
    Lpp = Lpp_
    [ch, n, m] = im.shape
    W = wei()
    div = 64 / math.sqrt(n * m)
    n *= div
    m *= div
    # LA TRANSFORMACION ES DIFERENTE EN PYTHON
    #imsSki = resize(im, (n, m), order=1, preserve_range=True)
    # LA TRANSFORMACION ES DIFERENTE EN PYTHON
    # VAMOS A PROBAR ESTA TRANSFORMACION
    ims = cv2.resize(im, (int(n), int(m)), interpolation=cv2.INTER_NEAREST)
    #np.savetxt("output.txt", ims)
    # otra transformacion --------- ims = ndimage.zoom(im, 21.4)
    #a = ndimage.zoom(im, n, prefilter=False)
    # VAMOS A PROBAR ESTA TRANSFORMACION
    # HAY QUE VER LAS TRASPUESTAS DE ESTAS MATRICES
    R = ims[0]
    G = ims[1]
    B = ims[2]
    # HAY QUE VER LAS TRASPUESTAS DE ESTAS MATRICE
    imV = [matrix_to_vcolumna(R), matrix_to_vcolumna(G), matrix_to_vcolumna(B)]
    # np.savetxt("output.txt", imV)
    t1 = np.random.permutation(imV[0].shape[0])
    # DEBERIA IR PERO NO SABEMOS ESPECIFICAMENTE QUE OCURRE EN Pg = [imV - imV(t1,:)];
    #Pg = imV
    ims = ndimage.zoom(ims, 0.5)
    #ims = cv2.resize(im, 0.5, interpolation=cv2.INTER_NEAREST)

    return ims


def wei():
    w = [0, 0, 1.0000, 0, 0.1000, 0.9000, 0, 0.2000, 0.8000, 0, 0.3000, 0.7000, 0, 0.4000, 0.6000, 0, 0.5000, 0.5000, 0,
         0.6000, 0.4000, 0, 0.7000, 0.3000, 0, 0.8000, 0.2000, 0, 0.9000, 0.1000, 0, 1.0000, 0, 0.1000, 0, 0.9000,
         0.1000, 0.1000, 0.8000, 0.1000, 0.2000, 0.7000, 0.1000, 0.3000, 0.6000, 0.1000, 0.4000, 0.5000, 0.1000, 0.5000,
         0.4000, 0.1000, 0.6000, 0.3000, 0.1000, 0.7000, 0.2000, 0.1000, 0.8000, 0.1000, 0.1000, 0.9000, 0, 0.2000, 0,
         0.8000, 0.2000, 0.1000, 0.7000, 0.2000, 0.2000, 0.6000, 0.2000, 0.3000, 0.5000, 0.2000, 0.4000, 0.4000, 0.2000,
         0.5000, 0.3000, 0.2000, 0.6000, 0.2000, 0.2000, 0.7000, 0.1000, 0.2000, 0.8000, 0, 0.3000, 0, 0.7000, 0.3000,
         0.1000, 0.6000, 0.3000, 0.2000, 0.5000, 0.3000, 0.3000, 0.4000, 0.3000, 0.4000, 0.3000, 0.3000, 0.5000, 0.2000,
         0.3000, 0.6000, 0.1000, 0.3000, 0.7000, 0.0000, 0.4000, 0, 0.6000, 0.4000, 0.1000, 0.5000, 0.4000, 0.2000,
         0.4000, 0.4000, 0.3000, 0.3000, 0.4000, 0.4000, 0.2000, 0.4000, 0.5000, 0.1000, 0.4000, 0.6000, 0.0000, 0.5000,
         0, 0.5000, 0.5000, 0.1000, 0.4000, 0.5000, 0.2000, 0.3000, 0.5000, 0.3000, 0.2000, 0.5000, 0.4000, 0.1000,
         0.5000, 0.5000, 0, 0.6000, 0, 0.4000, 0.6000, 0.1000, 0.3000, 0.6000, 0.2000, 0.2000, 0.6000, 0.3000, 0.1000,
         0.6000, 0.4000, 0.0000, 0.7000, 0, 0.3000, 0.7000, 0.1000, 0.2000, 0.7000, 0.2000, 0.1000, 0.7000, 0.3000,
         0.0000, 0.8000, 0, 0.2000, 0.8000, 0.1000, 0.1000, 0.8000, 0.2000, 0.0000, 0.9000, 0, 0.1000, 0.9000, 0.1000,
         0.0000, 1.0000, 0, 0]
    return w

# img = scipy.misc.imread('Paint3x3.png', 'RGB')
# print("Testing gcsdecolor")
# print("ESTA ES LA DE SCIPY")
# print(img)
# print(img.shape)
# print("ESTA ES LA DE MATPLOTLIB")
# f = plt.imread('Paint3x3.png')
# f = f.transpose()
# plt.imshow(f)
# print(f)
# print("--------SEPARADOR--------")
# print(f[0].transpose())
# print("--------SEPARADOR--------")
# print(f[1].transpose())
# print("--------SEPARADOR--------")
# print(f[2].transpose())
# print(f.shape)


# OPEN CV TEST

print("here1")
im = cv2.imread('PruebaPeppersRGB.png')
#im = formato_imagen_in('PruebaPeppersRGB.png')
imagen = gcsdecolor2python(im, 0.25)
print( "ancho: {} pixels" .format(imagen.shape[ 1 ]))
print( "alto: {} pixels" .format(imagen.shape[ 0 ]))
print( "canales: {} pixels" .format(imagen.shape[ 2 ]))
cv2.imshow("visor", imagen)
cv2.waitKey(0)
cv2.imshow("visor", im)
cv2.waitKey(0)
print("here2")
cv2.imwrite("nueva-imagen.jpg", im)
