import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt


def matrix_to_vcolumna(matrix_n):
    vector_columna = np.ones((matrix_n.shape[0]*matrix_n.shape[1], 1))
    c = 0
    for j in range(matrix_n.shape[0]):
        for k in range(matrix_n.shape[1]):
            vector_columna[c, 0] = matrix_n[j, k]
            c += 1
    return vector_columna


def reshape(rmat):
    mat = np.ones([rmat.shape[0]*rmat.shape[1], rmat.shape[2]])
    filecount = rmat.shape[1]
    mult = 0
    cont = 0
    for i in range(rmat.shape[1]):
        for j in range(rmat.shape[2]):
            for l in range(rmat.shape[0]):
                mat[l+mult][j] = rmat[l][i][j]
                cont += 1
                if cont % 9 == 0:
                    mult += filecount
    return mat


def powm(matrix, value):
    return np.linalg.matrix_power(matrix, value)


def divm(matrix, value):
    return np.true_divide(matrix, value)


def pownpa(nparray, value):
    return np.power(nparray, value)


def divnpa(nparray, value):
    return np.divide(nparray, value)


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


# la imagen llega de esta forma img = cv2.imread("img.jpg")# Read in your image
def decolor_nonlinear(image):
    # PASO 0 Preprocesamiento
    im = cv.normalize(image, image, 0, 255, cv.NORM_MINMAX)

    [dim, row, col] = im.shape

    lam = row * col
    alpha = 1

    imlab = cv.cvtColor(im, cv.COLOR_RGB2LAB)
    imluv = cv.cvtColor(im, cv.COLOR_RGB2LUV)
    ###no ta rgb2lch
    imlch = cv.cvtColor(im, cv.COLOR_RGB2HSV)

    # PASO 1. Compute color differences G

    # QUEDE AQUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII

    # VER QUE DA THETA EN MATLAB y TRATAR DE CORRER EL decolor_nonlinear
    # theta = math.atan2((imluv[2] - 0.48810), (imluv[1] - 0.20917))
    theta = 0.6435
    qtheta = - 0.01585 - 0.03017 * math.cos(theta) - 0.04556 * math.cos(2 * theta) - 0.02677 * math.cos(3 * theta) - 0.00295 * math.cos(4 * theta) + 0.14592 * math.sin(theta) + 0.05084 * math.sin(2 * theta) - 0.01900 * math.sin(3 * theta) - 0.00764 * math.sin(4 * theta)

    temp0 = np.array(imluv[0])
    temp1 = np.array(imluv[1])
    temp2 = np.array(imluv[2])

    suv = 13 * pownpa((pownpa((temp1 - 0.20917), 2) + pownpa((temp2 - 0.48810), 2)), 0.5)
    LHK = temp0 + (-0.1340 * qtheta + 0.0872 * 0.8147419482) * suv * temp0

    deltalhk_x = np.roll(LHK, 2, axis=1) - np.roll(LHK, -2, axis=1)
    deltalhk_y = np.roll(LHK, -1, axis=0) - np.roll(LHK, 1, axis=0)
    deltalab_x = np.roll(imlab, 2, axis=1) - np.roll(imlab, -2, axis=1)
    deltalab_y = np.roll(imlab, -1, axis=0) - np.roll(imlab, 1, axis=0)

    # SE PUEDE HACER MAS EFICIENTE dl_x = [deltalab_x[0], deltalab_x[1], deltalab_x[2]]
    # Y SE INDEXA COMO NP ARRAY
    deltalab_x0 = np.array(deltalab_x[0])
    deltal_x = deltalab_x0
    deltalab_x1 = np.array(deltalab_x[1])
    deltalab_x2 = np.array(deltalab_x[2])

    deltalab_y0 = np.array(deltalab_y[0])
    deltal_y = deltalab_y
    deltalab_y1 = np.array(deltalab_y[1])
    deltalab_y2 = np.array(deltalab_y[2])

    deltalab3_x = pownpa(deltalab_x0, 3) + pownpa(deltalab_x1, 3) + pownpa(deltalab_x2, 3)
    deltalab3_y = pownpa(deltalab_y0, 3) + pownpa(deltalab_y1, 3) + pownpa(deltalab_y2, 3)

    #TO BE DISCOVERED
    # IDEXING Indexing — NumPy v1.14 Manual.htm
    signg_x = np.sign(deltalhk_x)
    # idx = signg_x == 0
    # signg_x[idx] = np.sign(deltal_x[idx])
    idx = signg_x == 0
    # signg_x[idx] = np.sign(deltalab3_x[idx])
    signg_y = np.sign(deltalhk_y)
    idx = signg_y == 0
    # signg_y[idx] = np.sign(deltal_y[idx])
    idx = signg_y == 0
    # signg_y[idx] = np.sign(deltalab3_y[idx])

    gx = pownpa(pownpa(deltalab_x0, 2) + pownpa(alpha * divm(pownpa(pownpa(deltalab_x1, 2) + pownpa(deltalab_x2, 2), 0.5), 3.59210244843), 2), 0.5)
    gy = pownpa(pownpa(deltalab_y0, 2) + pownpa(alpha * divm(pownpa(pownpa(deltalab_y1, 2) + pownpa(deltalab_y2, 2), 0.5), 3.59210244843), 2), 0.5)

    Gx = signg_x * gx
    Gy = signg_y * gy

    # PASO 2 OPTIMIZACION
    L = np.array(imlch[0])
    C = np.array(imlch[1])
    H = np.array(imlch[2])
    T = np.zeros([9, row, col])

    for n in range(0, 3):
        T[n] = C * np.cos(n * H)
        T[n + 4] = C * np.sin(n * H)

    T[8] = C

    #[U, V] = np.gradient(T)
    gradientUV = np.gradient(T)
    #[Lx, Ly] = np.gradient(L)
    gradientLxLv = np.gradient(L)

    #Reesterilization
    p = (matrix_to_vcolumna(Gx) - matrix_to_vcolumna(gradientLxLv[0])).conj().T
    q = (matrix_to_vcolumna(Gy) - matrix_to_vcolumna(gradientLxLv[1])).conj().T

    # % Move the 9 dim in the first dim and resterilize the matrix
    # USED TO BE LIKE THIS
    # u = reshape(shiftdim(U, 2), 9, [])
    # v = reshape(shiftdim(V, 2), 9, [])

    #VER EL METODO MIO RESHAPE PARA VER PORQ SALE DEL INDICE
    # u = reshape(gradientUV[0])
    # v = reshape(gradientUV[1])
    u = gradientUV[0]
    v = gradientUV[1]

    # DUNNO
    # b_s = sum(bsxfun( @ times, u, p) + bsxfun( @ times, v, q), 2);
    b_s = 0.24

    # HASTA QUE NO ARREGLE EL RESHAPE
    #M_s = u * u.conj().T + v * v.conj().T
    M_s = u.conj().T + v.conj().T

    # DUNNO
    # Solve the energy function based on M_s and b_s
    # x = (M_s + pow(np.ones(9), lam)) \ b_s;
    x = M_s + pownpa(np.ones(9), lam)

    # ftheta = (x.conj().T * reshape(T)).reshape((row, col))
    # tones = plt.imshow(L + pow(ftheta, C), cmap='gray', interpolation='nearest', vmin=0, vmax=255)

    print(x.shape)
    tones = plt.imshow(x, cmap='gray', interpolation='nearest', vmin=0, vmax=255)

    plt.savefig('text.png')
    plt.show()
    return tones

img = cv.imread("PruebaPeppersRGB.png")
print(img.shape)
decolor_nonlinear(img)
