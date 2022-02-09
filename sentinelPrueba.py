from platform import platform
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date

#credenciales
api=SentinelAPI("desertificacion", "SaturdayAI", "https://scihub.copernicus.eu/dhus/#/home")

mascara=geojson_to_wkt(read_geojson('map.geojson'))
productos=api.query(mascara,date = ('20151219', date(2015, 12, 29)),
                     platformname = 'Sentinel-2',
                     cloudcoverpercentage = (0, 30))

#devuelve un orderedDict()
#print(productos)

#longitud
print(len(productos))
api.to_geojson(productos)
api.download_all(productos, directory_path=r'C:\data')
#api.get_product_odata('53eee050-0041-4767-81d4-bf071a82205e')

#api.download('53eee050-0041-4767-81d4-bf071a82205e')