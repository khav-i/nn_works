# Детекция масок

## Данные

Работа с Kaggle-датасетом [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/discussion/425708).

## Содержание работы и обоснование выбора моделей

Так же, как и при работе с данными VOC2012, здесь используются предобученные модели Faster R-CNN и YOLO.

На этот раз был использован `fastrcnn_resnet50_fpn_v2`, т.к. эта модель, как сказано в [документации](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html), создает улучшенную модель Faster R-CNN с магистральной сетью ResNet-50-FPN на основе сравнительного анализа передачи обучения с помощью Vision Transformers, что куда как лучше версии модели от 2015-го.

А использованная YOLO – m-версии 8-го поколения. Хотя недавно уже вышли модели 10-го поколения, их, насколько я знаю, еще не успели толком обкатать, и они, возможно, показывают чуть худший результат для объектов на дальней и на очень близких дистанциях.

А уж в сравнении с YOLOv5 плюсов предостаточно: и качество работы, и более удобный интерфейс.

## Сравнение обученных моделей

|model|epochs|time per epoch|1cls AP@.5|2cls AP@.5|3cls AP@.5|mAP@.5|
|-|-|-|-|-|-|-|
|Faster RCNN|30|2.0 min|0.5|0.893|0.791|0.728|
|YOLOv8|60|0.5 min|0.888|0.955|0.889|0.911|

Где:

* 1cls – `mask_weared_incorrect`;
* 2cls – `with_mask`
* 3cls – `without_mask`

Я счел нецелесообразным продлевать обучение Faster RCNN до 60 эпох, т.к. падение loss-а для валидационной выборки выходит на плато как раз в промежутке 20-30. Моих навыков для преодоления этой преграды пока что не хватает.

## Ссылки на тетрадки в обзорщике

[od_mask_recognition_faster_rcnn](https://nbviewer.org/github/khav-i/nn_works/blob/master/Face%20Mask%20Detection/od_mask_recognition_faster_rcnn.ipynb)

[od_mask_recognition_yolov8](https://nbviewer.org/github/khav-i/nn_works/blob/master/Face%20Mask%20Detection/od_mask_recognition_yolov8.ipynb)
