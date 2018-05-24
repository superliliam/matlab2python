import math
import numpy as np


def spdecolor(Im):
    spdecolor(Im, 0.005)


def spdecolor(Im, sigma):
    #si la imagen no tiene 3 dimensiones entonces es en escala de grises
    if Im.shape[2]!=3:
        grayIm = Im

    #parametros
    order = 2
    #BUSCAR ESTE INF si es Inf de infinito en matlab o es otra cosa
    pre_E = inf
    # BUSCAR ESTE INF si es Inf de infinito en matlab o es otra cosa

