# Детекция масок

## Данные

Работа с Kaggle-датасетом [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/discussion/425708).

## Содержание работы

Так же, как и при работе с данными VOC2012, здесь используются предобученные модели Faster RCNN и YOLO (на этот раз 8-го поколения m-версия).

## Сравнение моделей

|model|1cls AP@.5|2cls AP@.5|3cls AP@.5|mAP@.5|
|-|-|-|-|-|
|Faster RCNN|0.5|0.893|0.791|0.728|
|YOLOv8|0.888|0.955|0.889|0.911|

Где:

* 1 класс – это `mask_weared_incorrect`;
* 2 – `with_mask`
* 3 – `without_mask`

## Ссылки на тетрадки в обзорщике

[od_mask_recognition_faster_rcnn]()

[od_mask_recognition_yolov8]()
