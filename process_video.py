import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Импорт трекера SORT

# Конфигурация путей
INPUT_VIDEO = 'video/test.mp4'
OUTPUT_VIDEO = 'video/result.mp4'

# Инициализация модели и трекера
detection_model = YOLO('runs//detect//train2//weights//best.pt').to('cuda')
object_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.1)

def process_frame(frame, model):
    """
    Обрабатывает кадр: детектирует объекты, отслеживает их и визуализирует результаты
    """
    # Детектирование объектов
    detections = model(frame, conf=0.7, iou=0.5, agnostic_nms=True)
    
    for result in detections:
        # Извлечение данных детекции
        boxes = result.boxes.xyxy.cpu().numpy()
        conf_scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Формирование массива детекций [x1, y1, x2, y2, conf, class]
        detections_list = []
        for box, confidence, cls_id in zip(boxes, conf_scores, class_ids):
            detections_list.append(np.hstack((box, confidence, cls_id)))
        
        # Сортировка детекций по уверенности (по возрастанию)
        detections_list.sort(key=lambda x: x[4], reverse=False)
        detections_array = np.array(detections_list)
        
        # Подготовка данных для трекера [x1, y1, x2, y2, conf]
        tracker_input = detections_array[:, :5] if len(detections_array) > 0 else np.empty((0, 5))
        
        # Обновление трекера и обратный порядок треков
        tracked_objects = object_tracker.update(tracker_input)[::-1]
        
        # Визуализация результатов
        for idx, detection in enumerate(detections_array):
            x1, y1, x2, y2 = map(int, detection[:4])
            confidence, class_id = detection[4], int(detection[5])
            
            # Выбор цвета в зависимости от класса
            color = (255, 0, 0) if class_id == 1 else (0, 255, 0)
            
            # Отрисовка рамки
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Подпись с информацией
            label = f"conf: {confidence:.2f}, class: {class_id}, id: {tracked_objects[idx][4]}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def process_video(input_path, output_path, model):
    """
    Обрабатывает видео: читает кадры, применяет обработку, сохраняет результат
    """
    # Инициализация видеопотока
    video_capture = cv2.VideoCapture(input_path)
    if not video_capture.isOpened():
        raise IOError("Ошибка: Не удалось открыть видеофайл")
    
    # Получение параметров видео
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Инициализация видеозаписи
    video_writer = cv2.VideoWriter(
        output_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        fps, 
        (width, height)
    )
    
    # Обработка кадров
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        processed_frame = process_frame(frame, model)
        video_writer.write(processed_frame)
    
    # Освобождение ресурсов
    video_capture.release()
    video_writer.release()
    print("Обработка видео успешно завершена")

# Запуск обработки видео
if __name__ == "__main__":
    process_video(INPUT_VIDEO, OUTPUT_VIDEO, detection_model)