import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show


dataset = rasterio.open('images/LC08_L1TP_199031_20220117_20220123_02_T1_ndvi_M_[-5.3173828125,39.72831341029745,3.8232421875000004,42.879989517714826].tiff')


print(dataset.name)
print(dataset.mode)
print(dataset.closed)
print("*********************")
print(dataset.width)
print(dataset.height)
print(dataset.count)
print(dataset.crs)


show((dataset,1))