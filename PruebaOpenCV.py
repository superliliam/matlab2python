import cv2
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt

# Cargamos la imagen del disco duro
imagen = cv2.imread( "PruebaPeppersRGB.tiff" )
# Mostramos ancho, alto y canales
print( "ancho: {} pixels" .format(imagen.shape[ 1 ]))
print( "alto: {} pixels" .format(imagen.shape[ 0 ]))
print( "canales: {} pixels" .format(imagen.shape[ 2 ]))

# Mostramos la imagen en la pantalla
cv2.imshow( "visor" , imagen)
cv2.waitKey( 0 )

# Guardar la imagen con otro nombre y otro formato
# Solo hay que poner la extensión del formato que queramos guardar
cv2.imwrite( "nueva-imagen.jpg" ,imagen)