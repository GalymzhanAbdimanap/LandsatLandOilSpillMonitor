

from mmseg.apis import init_model, inference_model, show_result_pyplot
import rasterio
from PIL import Image
import cv2
import os
import numpy as np
import glob
import shutil
from fastapi import FastAPI
import uvicorn

app = FastAPI()

def process_tiff_multi_cl(tiff_path, output_png_path):

    # Читаем tiff-файл с использованием rasterio
    with rasterio.open(tiff_path) as src:
        # Читаем данные изображения
        image_array = src.read()  # Читаем все каналы
        nodata = src.nodata

    # Используем первые три канала для формирования RGB изображения
    # rgb_image = image_array[:3, :, :].transpose(1, 2, 0)
    rgb_image = image_array[[6, 3, 2], :, :].transpose(1, 2, 0)
    # Преобразуем тип данных к uint8
    rgb_image = (rgb_image * 255).astype(np.uint8)
    nodata_mask = (image_array[0] == 0)
    rgb_image[nodata_mask] = [50, 255, 198]
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

def predict(crop_images_path, out_predicted_folder):

    config_path = 'configs/mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py'
    checkpoint_path = 'configs/iter_40000.pth'

    model = init_model(config_path, checkpoint_path)

    filenames = glob.glob(f'{crop_images_path}/*.png')
    for filename in filenames:
        print(filename)
        img_path = filename
        result = inference_model(model, img_path)
        np.save(f'{out_predicted_folder}/{os.path.basename(filename)[:-3]}npy', result.pred_sem_seg.data.cpu().numpy(), allow_pickle=True, fix_imports=True)
        # vis_iamge = show_result_pyplot(model, img_path, result,  show=False,  out_file=f'data/landsat_test_images/images_320_320/{dir}/predictions_{model_name}/{os.path.basename(filename)}', save_dir='./') #draw_pred=False, draw_gt=False,  

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

def process(src_image_name, SRC_IMAGES_FOLDER = 'src_images', CROP_IMAGES_FOLDER = 'crop_images', PREDICTED_CROP_IMAGES_FOLDER = 'predicted_crop_images', PREDICTED_IMAGES_FOLDER = 'predicted_images'):

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
    result_predict = predict(result_crop_images, output_predicted_images_folder)


    # Combine predicted images.
    os.makedirs(PREDICTED_IMAGES_FOLDER, exist_ok=True)
    output_combine_path = os.path.join(PREDICTED_IMAGES_FOLDER, f'{os.path.basename(result_src_image)[:-4]}.npy')
    result_combine = combine_mask_images(result_predict, output_combine_path, SRC_IMAGES_FOLDER)

    # Convert rgb 2 rast.
    output_tif_path = os.path.join(PREDICTED_IMAGES_FOLDER, f'{os.path.basename(result_src_image)[:-4]}.tif')
    result_tif_file = save_tif(result_combine, src_image_name, output_tif_path)

    # Clear folders
    delete_files_and_folders(SRC_IMAGES_FOLDER)
    delete_files_and_folders(CROP_IMAGES_FOLDER)
    delete_files_and_folders(PREDICTED_CROP_IMAGES_FOLDER)
    delete_files_and_folders(PREDICTED_IMAGES_FOLDER, '.npy')


    return result_tif_file

@app.post("/segment")
def read_root(file_name: str):
    result_cl_file = process(file_name)
    return {"message": result_cl_file}



# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
# uvicorn main:app --host 172.20.107.6 --port 5544