# Tratamiento de datos
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio import plot


'''with open("salida2.txt", "r") as tf:
    lines = tf.read().split(' ')'''
tf=open('salida2.txt')
a=[]
'''data=tf.read()
data=data.rstrip()
a.append(data.rsplit())'''

for line in tf:
    #print((line.rstrip()).rsplit())
    a.append((line.rstrip()).rsplit())


Anp=np.array(a)

print(Anp[0][0])
print(Anp[0][8])
print(Anp[1][0])
#tamaño de las matrices a visualizar
size=(2000,2000)
fArray = Anp.astype(np.float64)
# Una matriz de ceros. 
imagen_negra = np.zeros(size)
filas=2000
columnas=2000

#Ynp=Anp[0,2000]
plot.show(fArray)
#visualizamos la matriz
#Se ve como una imagen negra, ya que todos los elementos (pixeles) tienen intensidad 0


