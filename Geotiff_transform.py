import rasterio
import numpy as np
import os
def transform_geotiff_data_type(tif_name,tran_type="uint16"):
    with rasterio.open(tif_name) as src:
        transform = src.transform
        height = src.height
        width = src.width
        count=src.count
        crs = src.crs
        dtype = src.dtypes
        matrix_data=src.read(1)
    if(dtype==tran_type):
        return
    # 创建新的 GeoTIFF 文件
    with rasterio.open(
            tif_name, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=tran_type,
            crs=crs,
            transform=transform) as dst:
        dst.write(matrix_data, 1)
    print(f"\n成功转换 GeoTIFF 文件{tif_name}")
if __name__=="__main__":
    tifs=os.listdir("F:\\2023")
    tifs=[tif for tif in tifs if tif.endswith(".tif")]
    for tif in tifs:
        transform_geotiff_data_type("F:\\2023\\"+tif)