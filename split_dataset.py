import os
import random
import shutil

def split_dataset(images_dir, labels_dir, output_dir, val_split=0.2, seed=42):
    """
    Разбивает датасет на тренировочную и валидационную выборки.

    Параметры:
    images_dir: путь к директории с исходными изображениями
    labels_dir: путь к директории с исходными аннотациями
    output_dir: корневая директория для выходных данных
    val_split: доля данных для валидации (по умолчанию 20%)
    seed: значение для инициализации генератора случайных чисел (для воспроизводимости)
    """
    
    random.seed(seed)

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Директория изображений не найдена: {images_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Директория аннотаций не найдена: {labels_dir}")

    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_lbl_dir = os.path.join(output_dir, 'train', 'labels')
    valid_img_dir = os.path.join(output_dir, 'valid', 'images')
    valid_lbl_dir = os.path.join(output_dir, 'valid', 'labels')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_lbl_dir, exist_ok=True)

    image_bases = set()
    for file in os.listdir(images_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(file)[0]
            image_bases.add(base_name)

    print(f"Найдено {len(image_bases)} изображений.")

    valid_files = []
    for base in image_bases:
        label_path = os.path.join(labels_dir, f"{base}.txt")
        if os.path.exists(label_path):
            valid_files.append(base)
        else:
            print(f"Предупреждение: Аннотация для {base}.txt не найдена")

    print(f"Действительных пар image-label: {len(valid_files)}")

    random.shuffle(valid_files)

    split_idx = int(len(valid_files) * (1 - val_split))
    train_files = valid_files[:split_idx]
    valid_files = valid_files[split_idx:]

    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}")

    def copy_files(bases, src_img, src_lbl, dst_img, dst_lbl):
        """
        Копирует изображения и соответствующие аннотации в целевые директории
        
        Параметры:
        bases: список базовых имен файлов (без расширений)
        src_img: исходная директория с изображениями
        src_lbl: исходная директория с аннотациями
        dst_img: целевая директория для изображений
        dst_lbl: целевая директория для аннотаций
        """
        for base in bases:
            img_copied = False
            for ext in ['.jpg', '.jpeg', '.png']:
                src_path = os.path.join(src_img, base + ext)
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_img)
                    img_copied = True
                    break
            
            lbl_src = os.path.join(src_lbl, base + '.txt')
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, dst_lbl)
            else:
                print(f"Ошибка: Аннотация {base}.txt отсутствует")

    print("Копирую train данные...")
    copy_files(
        train_files, 
        images_dir, 
        labels_dir,
        train_img_dir, 
        train_lbl_dir
    )
    
    print("Копирую valid данные...")
    copy_files(
        valid_files, 
        images_dir, 
        labels_dir,
        valid_img_dir, 
        valid_lbl_dir
    )

    print("Разделение датасета завершено успешно!")

if __name__ == '__main__':
    split_dataset(
        images_dir='data/images',
        labels_dir='data/labels',
        output_dir='data',
        val_split=0.2,
        seed=42
    )