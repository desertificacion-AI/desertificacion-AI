# Tratamiento de datos
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd

# Gráficos
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import sklearn

# Preprocesado y modelado
# ------------------------------------------------------------------------------
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor


#Necesitamos transformar la imagen tiff a una matriz de datos

from PIL import Image 
from numpy import asarray   
outfile_name='./out/T31TBG_20210316T105031_B_NDVI.tif' 
image1 = Image.open(outfile_name) 
outfile_name='./out/T31TBG_20210405T105021_B_NDVI.tif' 
image2 = Image.open(outfile_name) 
outfile_name='./out/T31TBG_20210505T105031_B_NDVI.tif' 
image3 = Image.open(outfile_name) 
outfile_name='./out/T31TBG_20210525T105031_B_NDVI.tif' 
image4 = Image.open(outfile_name) 
outfile_name='./out/T31TBG_20210614T105031_B_NDVI.tif' 
image5 = Image.open(outfile_name) 
outfile_name='./out/T31TBG_20210624T105031_B_NDVI.tif' 
image6 = Image.open(outfile_name) 
outfile_name='./out/T31TBG_20210714T105031_B_NDVI.tif'
#image7 = Image.open(outfile_name) 
    
'''print(image.format) 
print(image.size) 
print(image.mode)'''

numpydata1 = asarray(image1) 
numpydata2 = asarray(image2)
numpydata3 = asarray(image3)
numpydata4 = asarray(image4)
numpydata5 = asarray(image5)
numpydata6 = asarray(image6)
#numpydata7 = asarray(image7)
  
'''print(type(numpydata1))  
print(numpydata1.shape) 
print(numpydata1)
print(numpydata2)
print(numpydata3)
print(numpydata4)
print(numpydata5)
print(numpydata6)
print(numpydata7)'''

print ('x foto1 y (00):' , numpydata1[0,0])
print ('x foto1 y (01):' ,numpydata1[0,1])
print ('x foto1 y (02):' ,numpydata1[0,2])
print ('x foto1 y (10):' ,numpydata1[1,0])
print ('x foto1 y (11):' ,numpydata1[1,1])
print ('x foto1 y (11):' ,numpydata1[1,1])
print ('x foto1 y (12):' ,numpydata1[1,1])
print ('x foto1 y (20):' ,numpydata1[2,0])
print ('x foto1 y (21):' ,numpydata1[2,1])
print ('x foto1 y (22):', numpydata1[2,2])

X=[]
Y=[]
for i in range (3000,5000-2,3) :
    for j in range (3000,5000-2,3) :        
        '''X entrada
        Y salida'''
        X.append([numpydata1[i,j],numpydata1[i,j+1],numpydata1[i,j+2],
                 numpydata1[i+1,j],numpydata1[i+1,j+1],numpydata1[i+1,j+2],
                 numpydata1[i+2,j],numpydata1[i+2,j+1],numpydata1[i+2,j+2],
                 numpydata2[i,j],numpydata2[i,j+1],numpydata2[i,j+2],
                 numpydata2[i+1,j],numpydata2[i+1,j+1],numpydata2[i+1,j+2],
                 numpydata2[i+2,j],numpydata2[i+2,j+1],numpydata2[i+2,j+2],
                 numpydata3[i,j],numpydata3[i,j+1],numpydata3[i,j+2],
                 numpydata3[i+1,j],numpydata3[i+1,j+1],numpydata3[i+1,j+2],
                 numpydata3[i+2,j],numpydata3[i+2,j+1],numpydata3[i+2,j+2],
                 numpydata4[i,j],numpydata4[i,j+1],numpydata4[i,j+2],
                 numpydata4[i+1,j],numpydata4[i+1,j+1],numpydata4[i+1,j+2],
                 numpydata4[i+2,j],numpydata4[i+2,j+1],numpydata4[i+2,j+2],
                 numpydata5[i,j],numpydata5[i,j+1],numpydata5[i,j+2],
                 numpydata5[i+1,j],numpydata5[i+1,j+1],numpydata5[i+1,j+2],
                 numpydata5[i+2,j],numpydata5[i+2,j+1],numpydata5[i+2,j+2]                 
        ])
        Y.append([numpydata6[i,j],numpydata6[i,j+1],numpydata6[i,j+2],
                 numpydata6[i+1,j],numpydata6[i+1,j+1],numpydata6[i+1,j+2],
                 numpydata6[i+2,j],numpydata6[i+2,j+1],numpydata6[i+2,j+2]])

Xnp=np.array(X)
Ynp=np.array(Y)

np.savetxt('entrada2.txt', Xnp)
np.savetxt('salida2.txt',Ynp)

#Clasificador
cls=RandomForestRegressor()

cls.fit(X,Y)
cls.predict(X)

print(cls.predict(X))

