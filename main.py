import rasterio
from PIL import Image
import cv2
import os
import numpy as np
import glob
import shutil

from mmseg.apis import init_model, inference_model, show_result_pyplot

from fastapi import FastAPI
import uvicorn
from create_multichannel import extract_selected_files, create_multichannel_geotiff, crop_raster_by_shapefile
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = FastAPI()

def process_tiff_multi_cl(tiff_path, output_png_path):
    # Читаем tiff-файл с использованием rasterio
    with rasterio.open(tiff_path) as src:
        # Читаем данные изображения
        image_array = src.read()  # Читаем все каналы
        nodata = src.nodata

    # Нормализуем каждый канал
    normalized_array = np.zeros_like(image_array, dtype=np.float32)
    for i in range(image_array.shape[0]):
        channel = image_array[i]
        channel_min, channel_max = channel.min(), channel.max()
        # Избегаем деления на ноль, если все значения в канале одинаковы
        if channel_max > channel_min:
            normalized_array[i] = (channel - channel_min) / (channel_max - channel_min) * 255
        else:
            normalized_array[i] = channel  # Если канал константный, оставляем как есть

    # Используем первые три канала для формирования RGB изображения
    rgb_image = normalized_array[[2, 1, 0], :, :].transpose(1, 2, 0).astype(np.uint8)

    # Применяем маску для значений "nodata"
    nodata_mask = (image_array[0] == nodata) if nodata is not None else (image_array[0] == 0)
    rgb_image[nodata_mask] = [50, 255, 198]  # Цвет для "nodata"

    # Создаем объект изображения с использованием Pillow
    img = Image.fromarray(rgb_image)
    # Сохраняем изображение в формате PNG
    img.save(output_png_path, format='PNG')
    return output_png_path


def apply_sliding_window(image_path, window_size, stride, output_folder):
    # Загружаем изображение
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    output_filename = os.path.basename(image_path)[:-4]
    
    # Проходим по изображению с помощью скользящего окна
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Проверяем, не выходит ли окно за пределы изображения
            if x + window_size[0] > width:
                x = width - window_size[0]
            if y + window_size[1] > height:
                y = height - window_size[1]

            # Вырезаем кусок изображения
            window = img[y:y+window_size[1], x:x+window_size[0]]
            crop_filename = f'{output_folder}/{output_filename}_{x}_{y}.png'
            cv2.imwrite(crop_filename, window)

            # Выходим из цикла, если достигли нижнего правого угла изображения
            if y + window_size[1] >= height and x + window_size[0] >= width:
                break

    
    return output_folder

def predict(crop_images_path, out_predicted_folder, predicted_visual_folder):

    config_path = 'configs/mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py'
    checkpoint_path = 'configs/iter_40000.pth'

    model = init_model(config_path, checkpoint_path)

    filenames = glob.glob(f'{crop_images_path}/*.png')
    for filename in filenames:
        print(filename)
        img_path = filename
        result = inference_model(model, img_path)
        np.save(f'{out_predicted_folder}/{os.path.basename(filename)[:-3]}npy', result.pred_sem_seg.data.cpu().numpy(), allow_pickle=True, fix_imports=True)
        vis_iamge = show_result_pyplot(model, img_path, result,  show=False, with_labels=False, out_file=f'{predicted_visual_folder}/{os.path.basename(filename)}', save_dir='./') #draw_pred=False, draw_gt=False,  

    return out_predicted_folder



def combine_mask_images(input_folder, output_path, src_folder):
    # Получаем список всех файлов в папке input_folder
    mask_files = os.listdir(input_folder)

    
    # H, W = first_mask.shape
    # print(output_path)
    src_img = cv2.imread(f'{src_folder}/{os.path.basename(output_path)[:-4]}.png')
    H, W = src_img.shape[:2]
    # Создаем пустую маску, куда будем объединять другие маски
    combined_mask = np.zeros((H, W), dtype=np.uint8)
    
    # Проходимся по всем файлам масок
    for mask_file in mask_files:
        # Проверяем, что файл - маска
        if mask_file.endswith('.npy'):
            # Загружаем маску
            mask = np.load(f'{input_folder}/{mask_file}')
            mask = mask[0]
            # Извлекаем координаты x и y из названия файла
            x, y = map(int, mask_file[:-4].split('_')[-2:])
            

            try:
            # Добавляем изображение в общую матрицу по соответствующим координатам
                combined_mask[y:y+mask.shape[0], x:x+mask.shape[1]] = mask
            except:
                if x+mask.shape[1] > combined_mask.shape[1] and y + mask.shape[0] > combined_mask.shape[0]:
                    combined_mask[combined_mask.shape[0]-mask.shape[0]:combined_mask.shape[0], combined_mask.shape[1]-mask.shape[1]:combined_mask.shape[1]] = mask
                    
                elif x + mask.shape[0] > combined_mask.shape[1]:
                    combined_mask[y:y+mask.shape[0], combined_mask.shape[1]-mask.shape[1]:combined_mask.shape[1]] = mask 
                else:
                    combined_mask[combined_mask.shape[0]-mask.shape[0]:combined_mask.shape[0], x:x+mask.shape[1]] = mask


    np.save(output_path, combined_mask, allow_pickle=True, fix_imports=True)
    return output_path

def save_tif(input_path, tiff_path, output_filename):
    data = np.load(input_path)
    with rasterio.open(tiff_path) as src:
        # Чтение массива данных
        data_VV = src.read(1)
        transform = src.transform 
    # Указываем параметры для создания GeoTIFF файла
    
    count = 1  # Количество каналов
    height, width = data.shape  # Размеры массива
    dtype = data.dtype  # Тип данных массива

    # Открываем файл для записи с использованием контекстного менеджера
    with rasterio.open(output_filename, 'w', driver='GTiff', height=height, width=width, count=count, dtype=dtype, transform=transform) as dst:
        # Записываем массив в файл
        dst.write(data, indexes=1)
    
    return output_filename

def delete_files_and_folders(folder_path, file_extensions=None):
    """
    Удаляет все файлы и подпапки в указанной папке. 
    Если указаны расширения, удаляются только файлы с этими расширениями, остальные файлы и папки остаются.
    
    :param folder_path: Путь к папке, из которой удаляются файлы и подпапки.
    :param file_extensions: Расширение или список расширений для удаления. Если None — удаляются все файлы и подпапки.
    """
    # Удаляем все файлы и папки, если расширения не указаны
    if file_extensions is None:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f'Файл {item_path} был удален.')
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f'Папка {item_path} была удалена.')
            except Exception as e:
                print(f'Не удалось удалить {item_path}. Ошибка: {e}')
    else:
        # Если передано одно расширение, превращаем его в список
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]

        # Перебираем файлы и подпапки
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                # Если это файл с нужным расширением, удаляем его
                if os.path.isfile(item_path) and any(item.endswith(ext) for ext in file_extensions):
                    os.remove(item_path)
                    print(f'Файл {item_path} был удален.')
                # Если это подпапка, удаляем ее вместе с содержимым
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f'Папка {item_path} была удалена.')
            except Exception as e:
                print(f'Не удалось удалить {item_path}. Ошибка: {e}')

def combine_sliding_window_images(input_folder, output_path, src_folder):
    # Получаем список всех файлов в папке input_folder
    image_files = os.listdir(input_folder)
    src_img = cv2.imread(f'{src_folder}/{os.path.basename(output_path)[:-4]}.png')
    H, W = src_img.shape[:2]
    # Создаем пустое изображение, куда будем объединять другие изображения
    combined_image  = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Проходимся по всем файлам изображений
    for image_file in image_files:
        # Проверяем, что файл - изображение
        if image_file.endswith('.png'):
            # Загружаем изображение
            img = cv2.imread(os.path.join(input_folder, image_file))
            
            # Извлекаем координаты x и y из названия файла
            x, y = map(int, image_file[:-4].split('_')[-2:])
            # img = cv2.resize(img, (1250, 650))
            try:
            # Добавляем изображение в общую матрицу по соответствующим координатам
                combined_image[y:y+img.shape[0], x:x+img.shape[1]] = img
            except:
                if x+img.shape[1] > combined_image.shape[1] and y + img.shape[0] > combined_image.shape[0]:
                    combined_image[combined_image.shape[0]-img.shape[0]:combined_image.shape[0], combined_image.shape[1]-img.shape[1]:combined_image.shape[1]] = img
                    
                elif x + img.shape[0] > combined_image.shape[1]:
                    combined_image[y:y+img.shape[0], combined_image.shape[1]-img.shape[1]:combined_image.shape[1]] = img 
                else:
                    combined_image[combined_image.shape[0]-img.shape[0]:combined_image.shape[0], x:x+img.shape[1]] = img

    
    # Сохраняем объединенное изображение
    cv2.imwrite(output_path, combined_image)

def apply_geotransform_from_source(src_geotiff_path, input_image_path, output_geotiff_path):
    """
    Применяет геопривязку и систему координат из одного GeoTIFF к изображению и сохраняет его как новый GeoTIFF.
    
    Параметры:
    - src_geotiff_path: путь к GeoTIFF-файлу, откуда берется геопривязка и система координат.
    - input_image_path: путь к изображению, которое нужно привязать.
    - output_geotiff_path: путь для сохранения нового GeoTIFF с привязкой.
    """
    
    # Открываем исходный GeoTIFF для чтения геопривязки и CRS
    with rasterio.open(src_geotiff_path) as src:
        transform = src.transform
        crs = src.crs

    # Открываем изображение, которое нужно привязать, и считываем его данные
    with rasterio.open(input_image_path) as img:
        img_data = img.read(1)  # считываем данные первого канала (или другого, если есть)

    # Определяем параметры для создания нового GeoTIFF
    height, width = img_data.shape
    dtype = img_data.dtype

    # Создаем новый GeoTIFF с использованием геопривязки и CRS из исходного GeoTIFF
    with rasterio.open(
        output_geotiff_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(img_data, 1)

from osgeo import ogr, osr, gdal

def convert_geojson_with_gdal(input_file, output_file, source_srs="EPSG:32639", target_srs="EPSG:4326", attribute_filter="class = 1"):
    """
    Конвертирует входной GeoJSON в другой GeoJSON с указанием системы координат и фильтрацией по атрибуту.
    
    Параметры:
    - input_file: путь к исходному файлу GeoJSON.
    - output_file: путь для сохранения результата в формате GeoJSON.
    - source_srs: исходная система координат (например, "EPSG:32639").
    - target_srs: целевая система координат (например, "EPSG:4326").
    - attribute_filter: строка с фильтром по атрибуту (например, "class = 1").
    """
    
    # Открываем входной GeoJSON для чтения
    src_ds = ogr.Open(input_file)
    if not src_ds:
        raise ValueError(f"Не удалось открыть файл {input_file}")
    
    # Получаем первый слой данных из источника
    src_layer = src_ds.GetLayer()
    src_layer.SetAttributeFilter(attribute_filter)  # Устанавливаем фильтр по атрибуту

    # Устанавливаем исходную и целевую системы координат
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(int(source_srs.split(":")[1]))
    
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(int(target_srs.split(":")[1]))
    
    # Создаем трансформатор для преобразования координат
    coord_transform = osr.CoordinateTransformation(src_srs, tgt_srs)

    # Создаем выходной файл GeoJSON
    driver = ogr.GetDriverByName("GeoJSON")
    dst_ds = driver.CreateDataSource(output_file)
    # if not dst_ds:
    #     raise ValueError(f"Не удалось создать выходной файл {output_file}")
    # with driver.CreateDataSource(output_file) as dst_ds:
        # if not dst_ds:
        #     raise ValueError(f"Не удалось создать выходной файл {output_file}")
    
    
    # Создаем выходной слой с целевой системой координат
    dst_layer = dst_ds.CreateLayer("layer", srs=tgt_srs, geom_type=src_layer.GetGeomType())
    
    # Копируем схему атрибутов
    src_layer_def = src_layer.GetLayerDefn()
    for i in range(src_layer_def.GetFieldCount()):
        field_defn = src_layer_def.GetFieldDefn(i)
        dst_layer.CreateField(field_defn)
    
    # Копируем и трансформируем геометрию объектов
    dst_layer_def = dst_layer.GetLayerDefn()
    for feature in src_layer:
        geom = feature.GetGeometryRef()
        geom.Transform(coord_transform)  # Преобразуем координаты
        new_feature = ogr.Feature(dst_layer_def)
        new_feature.SetGeometry(geom)
        
        # Копируем значения атрибутов
        for i in range(dst_layer_def.GetFieldCount()):
            new_feature.SetField(dst_layer_def.GetFieldDefn(i).GetNameRef(), feature.GetField(i))
        
        dst_layer.CreateFeature(new_feature)
        new_feature = None
    
    # Закрываем файлы
    src_ds = None
    dst_ds = None
    print(f"Файл успешно конвертирован и сохранен в {output_file}.")

    return output_file

def raster_to_vector(input_raster, output_geojson):
    """
    Преобразует одноканальный растровый файл в векторный формат GeoJSON с полигонализацией классов.
    
    Parameters:
    input_raster (str): Путь к входному растровому файлу (GeoTIFF).
    output_geojson (str): Путь к выходному векторному файлу (GeoJSON).
    """
    # Открываем растровый файл
    src_ds = gdal.Open(input_raster)
    if src_ds is None:
        raise FileNotFoundError(f"Не удалось открыть растровый файл: {input_raster}")
    
    band = src_ds.GetRasterBand(1)  # Предполагается, что у вас один канал
    
    # Создаём векторный файл GeoJSON
    driver = ogr.GetDriverByName("GeoJSON")
    out_ds = driver.CreateDataSource(output_geojson)
    if out_ds is None:
        raise RuntimeError(f"Не удалось создать векторный файл: {output_geojson}")
    
    # Создаём слой для полигонов
    out_layer = out_ds.CreateLayer("polygonized", geom_type=ogr.wkbPolygon)
    
    # Добавляем поле для значений класса
    field = ogr.FieldDefn("class", ogr.OFTInteger)
    out_layer.CreateField(field)
    
    # Полигонализация растра
    gdal.Polygonize(band, None, out_layer, 0, [], callback=None)
    
    # Закрываем файлы
    src_ds = None
    out_ds = None
    
    print(f"Полигональный векторный файл создан: {output_geojson}")
    return output_geojson


def process(src_image_name, SRC_IMAGES_FOLDER = 'src_images', CROP_IMAGES_FOLDER = 'crop_images', PREDICTED_CROP_IMAGES_FOLDER = 'predicted_crop_images', PREDICTED_IMAGES_FOLDER = 'predicted_images', PREDICTED_VISUALIZATION_FOLDER='visualization_crop'):

    # Convert rastr 2 rgb.
    os.makedirs(SRC_IMAGES_FOLDER, exist_ok=True)
    output_png_path = os.path.join(SRC_IMAGES_FOLDER, f'{os.path.basename(src_image_name)[:-4]}.png')
    result_src_image = process_tiff_multi_cl(src_image_name, output_png_path)

    # Crop src image to fragments(320x320)
    window_size = (320, 320)
    stride = 320
    output_crop_images_folder = os.path.basename(result_src_image)[:-4]
    os.makedirs(CROP_IMAGES_FOLDER, exist_ok=True)
    output_crop_images_folder = os.path.join(CROP_IMAGES_FOLDER, output_crop_images_folder)
    os.makedirs(output_crop_images_folder, exist_ok=True)
    result_crop_images = apply_sliding_window(result_src_image, window_size, stride, output_crop_images_folder)
    
    # Predict images.
    os.makedirs(PREDICTED_CROP_IMAGES_FOLDER, exist_ok=True)
    output_predicted_images_folder = os.path.join(PREDICTED_CROP_IMAGES_FOLDER, os.path.basename(result_crop_images))
    os.makedirs(output_predicted_images_folder, exist_ok=True)
    os.makedirs(PREDICTED_VISUALIZATION_FOLDER, exist_ok=True)
    result_predict = predict(result_crop_images, output_predicted_images_folder, PREDICTED_VISUALIZATION_FOLDER)


    # Combine predicted images.
    os.makedirs(PREDICTED_IMAGES_FOLDER, exist_ok=True)
    output_combine_path = os.path.join(PREDICTED_IMAGES_FOLDER, f'{os.path.basename(result_src_image)[:-4]}.npy')
    image_combine_path = os.path.join(PREDICTED_IMAGES_FOLDER, f'{os.path.basename(result_src_image)[:-4]}.png')
    result_combine = combine_mask_images(result_predict, output_combine_path, SRC_IMAGES_FOLDER)
    image_combine = combine_sliding_window_images(PREDICTED_VISUALIZATION_FOLDER, image_combine_path, SRC_IMAGES_FOLDER)

    # Convert rgb 2 rast.
    output_tif_path = os.path.join(PREDICTED_IMAGES_FOLDER, f'{os.path.basename(result_src_image)[:-4]}.tif')
    result_tif_file = save_tif(result_combine, src_image_name, output_tif_path)

    src_geotiff_path = src_image_name
    input_image_path = result_src_image
    output_geotiff_path = os.path.join(PREDICTED_IMAGES_FOLDER, f'{os.path.basename(result_src_image)[:-4]}_img.tif')
    apply_geotransform_from_source(src_geotiff_path, input_image_path, output_geotiff_path)

    # Clear folders
    delete_files_and_folders(SRC_IMAGES_FOLDER)
    delete_files_and_folders(CROP_IMAGES_FOLDER)
    delete_files_and_folders(PREDICTED_CROP_IMAGES_FOLDER)
    delete_files_and_folders(PREDICTED_IMAGES_FOLDER, '.npy')
    delete_files_and_folders(PREDICTED_VISUALIZATION_FOLDER)


    return result_tif_file


def run(folder='land_new'):

    files = os.listdir(folder)
    out_extract_folder = 'src_tif_files'
    out_multichannel_folder = 'src_inputs'
    out_crop_folder = 'inputs'
    os.makedirs(out_extract_folder, exist_ok=True)
    os.makedirs(out_multichannel_folder, exist_ok=True)
    os.makedirs(out_crop_folder, exist_ok=True)
    for file in files:
        try:
            tar_file = os.path.join(folder, file)
            extract_files = [f"{file[:-4]}_SR_B4.TIF", f"{file[:-4]}_SR_B5.TIF", f"{file[:-4]}_SR_B7.TIF"]
            extract_selected_files(tar_file, extract_files, out_extract_folder)
            out_path = os.path.join(out_multichannel_folder, f"{file[:-4]}.tif")
            create_multichannel_geotiff(file[:-4], out_extract_folder, out_path)
            raster_path = out_path
            shapefile_path = 'configs/OMG_area.shp'
            crop_out_path = os.path.join(out_crop_folder,  f"{file[:-4]}.tif")
            crop_raster_by_shapefile(raster_path, shapefile_path, crop_out_path)
            result_cl_file = process(crop_out_path)

            delete_files_and_folders(out_extract_folder)
            delete_files_and_folders(out_multichannel_folder)
            

            print(f"Обработка файла {crop_out_path} закончилась, результат по адресу {result_cl_file}")
        except Exception as e:
            print(f'Ошибка при обработке файла {file}')
    return "Обработка файлов закончена, результаты в папке predicted_images"

@app.post("/segment")
def read_root(folder_name: str):
    result = run(folder_name)

    return {"message": result}



# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
