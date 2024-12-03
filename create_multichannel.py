import rasterio
import numpy as np
import os
import tarfile
import geopandas as gpd
from rasterio.mask import mask

def crop_raster_by_shapefile(raster_path, shapefile_path, output_path):
    """
    Обрезает .tif файл по шейп-файлу и сохраняет результат в новый файл.
    
    Параметры:
        raster_path (str): Путь к исходному .tif файлу.
        shapefile_path (str): Путь к шейп-файлу (например, с полигонами).
        output_path (str): Путь для сохранения обрезанного .tif файла.
    """
    # Загружаем шейп-файл с использованием geopandas
    shapefile = gpd.read_file(shapefile_path)
    
    # Открываем растровый файл с использованием rasterio
    with rasterio.open(raster_path) as src:
        # Преобразуем геометрию шейп-файла в формат, понятный rasterio
        geoms = shapefile.geometry.values  # Извлекаем геометрию (полигоны)
        
        # Обрезаем растровые данные по геометрии шейп-файла
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta
        
        # Обновляем метаданные для сохранения
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "count": out_image.shape[0]  # Указываем количество каналов
        })
        
        # Сохраняем обрезанный .tif файл
        with rasterio.open(output_path, "w", **out_meta) as dest:
            for i in range(out_image.shape[0]):  # Записываем каждый канал отдельно
                dest.write(out_image[i], i + 1)
    
    print(f"Обрезанный .tif файл сохранен как: {output_path}")

def extract_selected_files(tar_file, files_to_extract, extract_to):
    """
    Извлекает выбранные файлы из tar-архива в указанную директорию.
    
    Параметры:
        tar_file (str): Путь к .tar файлу.
        files_to_extract (list): Список имен файлов для извлечения.
        extract_to (str): Путь к директории для извлечения файлов.
    """
    # Проверка, существует ли директория для извлечения
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    # Открываем .tar файл
    with tarfile.open(tar_file, "r") as tar:
        # Перебираем файлы в архиве
        for member in tar.getmembers():
            # Проверяем, есть ли файл в списке для извлечения
            if member.name in files_to_extract:
                tar.extract(member, path=extract_to)
                print(f"Извлечен файл: {member.name}")
    
    print(f"Файлы извлечены в: {extract_to}")

def create_multichannel_geotiff(input_file, root_dir, output_file):
    """
    Создает многоканальный GeoTIFF из списка одноканальных GeoTIFF файлов.

    Параметры:
        input_files (list): Список имен входных файлов GeoTIFF.
        root_dir (str): Путь к директории, где находятся входные файлы.
        output_file (str): Имя выходного файла GeoTIFF.
    
    Возвращает:
        str: Путь к созданному многоканальному GeoTIFF файлу.
    """
    # Чтение всех входных файлов
    input_files = [f"{input_file}_SR_B4.tif", f"{input_file}_SR_B5.tif", f"{input_file}_SR_B7.tif"]
    datasets = [rasterio.open(os.path.join(root_dir, file)) for file in input_files]

    # Проверка на соответствие геопозиции
    for ds in datasets[1:]:
        if ds.transform != datasets[0].transform or ds.crs != datasets[0].crs:
            raise ValueError("Все входные файлы должны иметь одинаковую геопозицию и CRS!")

    # Создание массива для данных всех каналов
    channels = [ds.read(1) for ds in datasets]

    # Объединение каналов
    multichannel_array = np.stack(channels, axis=0)

    # Запись многоканального GeoTIFF
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=datasets[0].height,
        width=datasets[0].width,
        count=len(channels),
        dtype=datasets[0].dtypes[0],
        crs=datasets[0].crs,
        transform=datasets[0].transform
    ) as dst:
        for i, channel in enumerate(multichannel_array, start=1):
            dst.write(channel, i)

    # Закрытие всех открытых файлов
    for ds in datasets:
        ds.close()

    print(f"Многоканальный GeoTIFF успешно создан: {output_file}")
    return output_file












# # Список файлов и порядок каналов
# # input_files = ["LC08_L2SP_165030_20241124_20241127_02_T1_SR_B2.tif", "LC08_L2SP_165030_20241124_20241127_02_T1_SR_B3.tif", "LC08_L2SP_165030_20241124_20241127_02_T1_SR_B4.tif", "LC08_L2SP_165030_20241124_20241127_02_T1_SR_B5.tif", "LC08_L2SP_165030_20241124_20241127_02_T1_SR_B6.tif", "LC08_L2SP_165030_20241124_20241127_02_T1_ST_B10.tif", "LC08_L2SP_165030_20241124_20241127_02_T1_SR_B7.tif"]
# input_files = ["LC08_L2SP_165030_20241124_20241127_02_T1_SR_B4.tif", "LC08_L2SP_165030_20241124_20241127_02_T1_SR_B5.tif", "LC08_L2SP_165030_20241124_20241127_02_T1_SR_B7.tif"]

# output_file = "output_multichannel.tif"
# root = '../../land_new'
# # Чтение всех входных файлов
# datasets = [rasterio.open(os.path.join(root,file)) for file in input_files]

# # Проверка на соответствие геопозиции
# for ds in datasets[1:]:
#     if ds.transform != datasets[0].transform or ds.crs != datasets[0].crs:
#         raise ValueError("Все входные файлы должны иметь одинаковую геопозицию и CRS!")

# # Создание массива для данных всех каналов
# channels = [ds.read(1) for ds in datasets]

# # Объединение каналов
# multichannel_array = np.stack(channels, axis=0)

# # Запись многоканального GeoTIFF
# with rasterio.open(
#     output_file,
#     'w',
#     driver='GTiff',
#     height=datasets[0].height,
#     width=datasets[0].width,
#     count=len(channels),
#     dtype=datasets[0].dtypes[0],
#     crs=datasets[0].crs,
#     transform=datasets[0].transform
# ) as dst:
#     for i, channel in enumerate(multichannel_array, start=1):
#         dst.write(channel, i)

# print(f"Многоканальный GeoTIFF успешно создан: {output_file}")
