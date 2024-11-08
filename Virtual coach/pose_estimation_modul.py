import os
import re
import math
import numpy as np
from io import BytesIO
from tqdm import tqdm
from collections import OrderedDict

from scipy.spatial import KDTree

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision import models

import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Определение трансформации
transform = T.Compose([T.ToTensor()])

keypointrcnn = models.detection.keypointrcnn_resnet50_fpn(weights='COCO_LEGACY')

# --------------------------------------------------------------------------


def crop_and_pad(image, target_size):
    """
    Обрезает или дополняет изображение до заданного размера.

    Эта функция принимат изображение и целевой размер, и в зависимости
    от размеров изображения выполняет одну из двух операций:
    1. Если размеры изображения превышают целевой размер, изображение будет
       обрезано до целевого размера, сохраняя центральную часть.
    2. Если размеры изображения меньше целевого размера, изображение будет
    дополнено пустыми областями (черным фоном) до целевого размера.

    Параметры:
    ----------
    image : numpy.ndarray 
        Входное изображение в формате (высота, ширина, каналы),
        где каналы обычно представляют собой цветовые компоненты (например, RGB).
        
    target_size : tuple
        Целевой размер изображения в формате (высота, ширина). Ожидается,
        что значения будут положительными целыми числами.

    Возвращает:
    ----------
    numpy.ndarray Изображение, обрезанное или дополненное до целевого размера
    в формате (высота, ширина, каналы).

    Примечания:
    ----------
    - Входное изображение должно быть в формате numpy.ndarray с 3 каналами.
    - Если входное изображение имеет размеры, равные целевым, оно будет
      возвращено без изменений.
    - Дополняемое изображение заполняется нулями (черным цветом).
    
    Примеры:
    --------
    >>> import numpy as np
    >>> img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    >>> resized_img = crop_and_pad(img, (150, 150))
    >>> print(resized_img.shape)
    (150, 150, 3)
    """
    # Получаем текущий размер изображения
    h, w, _ = image.shape
    
    # Если размер больше заданного, обрезаем
    if (h > target_size[0]) or (w > target_size[1]):
        y_start = (h - target_size[0]) // 2
        x_start = (w - target_size[1]) // 2
        image = image[y_start:y_start + target_size[0], x_start:x_start + target_size[1]]
    else:
        # Если размер меньше заданного, дополняем пустыми областями
        # Создаем изображение с фоном (черным или другим цветом)
        padded_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        
        # Расчет координат размещения оригинального изображения в новом
        y_offset = (target_size[0] - h) // 2
        x_offset = (target_size[1] - w) // 2
        
        # Копируем оригинальное изображение в новое с учетом смещения
        padded_image[y_offset:y_offset + h, x_offset:x_offset + w] = image
        image = padded_image

    return image


# --------------------------------------------------------------------------


def show(imgs):
    """
    Отображает одно или несколько изображений с использованием Matplotlib.

    Эта функция принимает одно изображение или список изображений и отображает 
    их в виде подгруппы с использованием Matplotlib. Если передано одно изображение,
    оно будет преобразовано в список для унификации обработки.

    Параметры:
    ----------
    imgs : Union[numpy.ndarray, list]
        Одно изображение в формате numpy.ndarray или список изображений. 
        Изображения должны быть в формате, совместимом с библиотекой PyTorch 
        (например, тензоры).

    Примечания:
    ----------
    - Изображения преобразуются в формат PIL перед отображением.
    - Оси (метки и деления) отключены для лучшего визуального восприятия.
    - Для работы функции требуется библиотека Matplotlib и torchvision.transforms.functional.

    Пример:
    --------
    >>> import torch
    >>> import torchvision.transforms.functional as F
    >>> import matplotlib.pyplot as plt 
    >>> img1 = torch.rand(3, 256, 256)  # Пример случайного тензора изображения
    >>> img2 = torch.rand(3, 256, 256)  # Второе изображение
    >>> show([img1, img2])
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
        
# --------------------------------------------------------------------------


def draw_keypoints_per_person(
    img, all_keypoints, all_scores, confs, boxes, keypoint_threshold=2, conf_threshold=0.9
):
    """
    Рисует ключевые точки и ограничивающие рамки 
    для каждого обнаруженного человека на изображении.

    Эта функция принимает изображение и соответствующие данные о
    ключевых точках, оценках уверенности, ограничивающих рамках и
    рисует их на изображении. Ключевые точки и рамки рисуются только для людей, 
    чья уверенность превышает заданный порог.

    Параметры:
    ----------
    img : numpy.ndarray 
        Входное изображение, на котором будут рисоваться ключевые точки и рамки.
        Ожидается, что изображение будет в формате (высота, ширина, каналы).
    all_keypoints : numpy.ndarray 
        Массив всех ключевых точек для обнаруженных людей, 
        размером (количество людей, количество ключевых точек, 2).
    all_scores : numpy.ndarray 
        Массив оценок уверенности для каждой ключевой точки, размером 
        (количество людей, количество ключевых точек).
        
    confs : list
        Список оценок уверенности для каждого человека, размером (количество людей).
        
    boxes : numpy.ndarray 
        Массив ограничивающих рамок для каждого человека,
        размером (количество людей, 4), где 4 - это (x1, y1, x2, y2) координаты.
        
    keypoint_threshold : float, optional
        Порог уверенности для рисования ключевых точек. По умолчанию 2.
        
    conf_threshold : float, optional
        Порог уверенности для фильтрации людей. По умолчанию 0.9.

    Возвращает:
    ----------
    numpy.ndarray
        Изображение с нарисованными ключевыми точками и ограничивающими рамками.

    Примечания:
    ----------
    - Использует цветовую карту HSV для генерации уникальных цветов для каждого человека.
    - Убедитесь, что входные данные имеют правильные размеры и типы, чтобы избежать ошибок.

    Пример:
    --------
    >>> import numpy as np
    >>> import cv2 
    >>> img = np.zeros((480, 640, 3), dtype=np.uint8)  # Пример пустого изображения 
    >>> all_keypoints = np.random.rand(2, 17, 2)  # Пример случайных ключевых точек для 2 людей
    >>> all_scores = np.random.rand(2, 17)  # Пример случайных оценок уверенности
    >>> confs = [0.95, 0.85]  # Уверенность двух людей
    >>> boxes = np.array([[100, 100, 300, 300], [400, 100, 600, 300]])  # Ограничивающие рамки
    >>> result_img = draw_keypoints_per_person(img, all_keypoints, all_scores, confs, boxes)
    """
    # создаем спектр цветов с помощью hsv
    cmap = plt.get_cmap("hsv")
    # создаем копию изображения 
    img_copy = np.array(img.copy())
    # Фильтруем людей по уверенности 
    valid_people = [i for i in range(len(all_keypoints)) if confs[i] > conf_threshold]
    num_people = len(valid_people)
    
    # Генерируем цвета только для людей с достаточной уверенностью
    colors = [tuple(np.asarray(cmap(i / num_people)[:-1]) * 255) for i in range(num_people)]
    
    # для каждого задетектированного человека
    for idx, person_id in enumerate(valid_people):
        # собираем опорные точки конкретного человека 
        keypoints = all_keypoints[person_id, ...]
        scores = all_scores[person_id, ...]
        # Рисуем ограничивающую рамку
        box = boxes[person_id]
        x1, y1, x2, y2 = map(int, box.detach().numpy().tolist())
        box_color = colors[idx]  # Используем цвет для рамки 
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), box_color, 2)  # Рисуем рамку
        
        # итерируем по каждому скору 
        for kp in range(len(scores)):
            # проверяем степень уверенности детектора опорной точки 
            if scores[kp] > keypoint_threshold:
                # конвертируем массив опорных точек в список целых чисел
                keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                keypoint_color = colors[idx]  # Используем тот же цвет для ключевой точки 
                # рисуем кружок радиуса 3 вокруг точки
                cv2.circle(img_copy, keypoint, 3, keypoint_color, -1)

    return img_copy


# --------------------------------------------------------------------------


def get_limbs_from_keypoints(keypoints):
    """
    Получает пары ключевых точек, представляющих конечности
    на основе заданного списка ключевых точек.

    Эта функция принимает список названий ключевых точек и
    возвращает список пар индексов, которые представляют
    конечности тела. Каждая пара соответствует связи между
    двумя ключевыми точками, что позволяе визуализировать
    скелетное представление человека.

    Параметры:
    ----------
    keypoints : list 
        Список строк, представляющих названия ключевых точек.
        Ожидается, что список будет содержать следующие ключевые точки:
        - "right_eye"
        - "left_eye"
        - "nose"
        - "right_ear"
        - "left_ear"
        - "right_shoulder"
        - "left_shoulder"
        - "right_elbow"
        - "left_elbow"
        - "right_wrist"
        - "left_wrist"
        - "right_hip"
        - "left_hip"
        - "right_knee"
        - "left_knee"
        - "right_ankle"
        - "left_ankle"

    Возвращает:
    ----------
    list
        Список пар индексов, где каждая пара представляет связь
        между двумя ключевыми точками (конечностями).
        Каждая пара является списком из двух индексов.

    Примечания:
    ----------
    - Функция предполагает, что все ключевые точки присутствуют 
      в переданном списке. Если какой-либо из элементов отсутствует,
      будет вызвано исключение `ValueError`.
    - Пары конечностей могут быть использованы для визуализации
      или анализа позы человека.

    Пример:
    --------
    >>> keypoints = [
    ...     "right_eye", "left_eye", "nose", "right_ear", "left_ear",
    ...     "right_shoulder", "left_shoulder", "right_elbow", "left_elbow",
    ...     "right_wrist", "left_wrist", "right_hip", "left_hip",
    ...     "right_knee", "left_knee", "right_ankle", "left_ankle"
    ... ]
    >>> limbs = get_limbs_from_keypoints(keypoints)
    >>> print(limbs)
    [
        [1, 2], [1, 3], [0, 2], [0, 4], 
        [5, 6], [6, 7], [5, 8], [8, 9],
        [11, 12], [12, 13], [10, 12], [13, 14],
        [5, 6], [10, 11], [5, 10], [6, 11]
    ]
    """
    limbs = [
        [keypoints.index("right_eye"), keypoints.index("nose")],
        [keypoints.index("right_eye"), keypoints.index("right_ear")],
        [keypoints.index("left_eye"), keypoints.index("nose")],
        [keypoints.index("left_eye"), keypoints.index("left_ear")],
        [keypoints.index("right_shoulder"), keypoints.index("right_elbow")],
        [keypoints.index("right_elbow"), keypoints.index("right_wrist")],
        [keypoints.index("left_shoulder"), keypoints.index("left_elbow")],
        [keypoints.index("left_elbow"), keypoints.index("left_wrist")],
        [keypoints.index("right_hip"), keypoints.index("right_knee")],
        [keypoints.index("right_knee"), keypoints.index("right_ankle")],
        [keypoints.index("left_hip"), keypoints.index("left_knee")],
        [keypoints.index("left_knee"), keypoints.index("left_ankle")],
        [keypoints.index("right_shoulder"), keypoints.index("left_shoulder")],
        [keypoints.index("right_hip"), keypoints.index("left_hip")],
        [keypoints.index("right_shoulder"), keypoints.index("right_hip")],
        [keypoints.index("left_shoulder"), keypoints.index("left_hip")],
    ]
    return limbs


coco_keypoints = [
    'nose','left_eye','right_eye',
    'left_ear','right_ear','left_shoulder',
    'right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip',
    'right_hip','left_knee', 'right_knee', 
    'left_ankle','right_ankle'
]

limbs = get_limbs_from_keypoints(coco_keypoints)

# --------------------------------------------------------------------------


def draw_skeleton_per_person(
    img, all_keypoints, all_scores, confs, limbs,
    boxes, keypoint_threshold=2, conf_threshold=0.9
    ):
    """
    Рисует скелетное представление каждого обнаруженного человека
    на изображении, используя ключевые точки и их уверенности.

    Эта функция принимает изображение и данные о ключевых точках,
    оценках уверенности, ограничивающих рамках и рисует скелетное
    представление для каждого человека, чья уверенность превышает
    заданный порог.

    Параметры:
    ----------
    img : numpy.ndarray 
        Входное изображение, на котором будет нарисовано скелетное
        представление. Ожидается, что изображение будет в формате 
        (высота, ширина, каналы).

    all_keypoints : numpy.ndarray 
        Массив всех ключевых точек для обнаруженных людей, размером
        (количество людей, количество ключевых точек, 2).

    all_scores : numpy.ndarray
        Массив оценок уверенности для каждой ключевой точки, размером
        (количество людей, количество ключевых точек).

    confs : list 
        Список оценок уверенности для каждого человека, 
        размером (количество людей).

    limbs : list
        Список пар индексов, представляющих конечности, где каждая
        пара указывает на индексы ключевых точек, соединенных линией.

    boxes : numpy.ndarray 
        Массив ограничивающих рамок для каждого человека, размером
        (количество людей, 4), где 4 - это (x1, y1, x2, y2) координаты.

    keypoint_threshold : float, optional
        Порог уверенности для рисования конечностей. По умолчанию 2.

    conf_threshold : float, optional 
        Порог уверенности для фильтрации людей. По умолчанию 0.9.

    Возвращает:
    ----------
    tuple
        - img_copy : numpy.ndarray 
            Изображение с нарисованным скелетом для каждого человека.
        - points : OrderedDict
            Словарь, содержащий координаты ключевых точек и 
            соответствующие им индексы конечностей.
        - vectors : list
            Список векторов, представляющих линии между конечностями.
        - vectors_scores : list
            Список оценок уверенности для каждой конечности.

    Примечания:
    ----------
    - Использует цветовую карту HSV для генерации 
      уникальных цветов для каждого человека.
    - Убедитесь, что входные данные имеют правильные 
      размеры и типы, чтобы избежать ошибок.

    Пример:
    --------
    >>> import numpy as np
    >>> import cv2
    >>> img = np.zeros((480, 640, 3), dtype=np.uint8)  # Пример пустого изображения
    >>> all_keypoints = np.random.rand(2, 17, 2)  # Пример случайных ключевых точек для 2 людей
    >>> all_scores = np.random.rand(2, 17)  # Пример случайных оценок уверенности
    >>> confs = [0.95, 0.85]  # Уверенность двух людей
    >>> boxes = np.array([[100, 100, 300, 300], [400, 100, 600, 300]])  # Ограничивающие рамки
    >>> limbs = get_limbs_from_keypoints([
    ... "right_eye", "left_eye", "nose", "right_ear", "left_ear",
    ... "right_shoulder", "left_shoulder", "right_elbow", "left_elbow",
    ... "right_wrist", "left_wrist", "right_hip", "left_hip",
    ... "right_knee", "left_knee", "right_ankle", "left_ankle"])  # Получаем конечности
    >>> [result_img, points, 
    ... vectors, vectors_scores] = draw_skeleton_per_person(
    ... img, all_keypoints, all_scores, confs, limbs, boxes
    ... )
    """
    cmap = plt.get_cmap("hsv")
    img_copy = np.array(img.copy())
    points = OrderedDict()
    vectors = []
    vectors_scores = []

    # Фильтрация людей по уверенности
    valid_people = [i for i in range(len(all_keypoints)) if confs[i] > conf_threshold]
    num_people = len(valid_people)
    
    # Генерация цветов только для людей с достаточной уверенностью
    colors = [tuple(np.asarray(cmap(i / num_people)[:-1]) * 255) for i in range(num_people)]

    if len(all_keypoints) > 0:
        for idx, person_id in enumerate(valid_people):
            keypoints = all_keypoints[person_id, ...]
            # Рисуем ограничивающую рамку
            box = boxes[person_id]
            x1, y1, x2, y2 = map(int, box.detach().numpy().tolist())
            box_color = colors[idx]  # Используем цвет для рамки 
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), box_color, 2)  # Рисуем рамку

            for limb_id in range(len(limbs)):
                limb_loc1 = keypoints[limbs[limb_id][0], :2].detach().numpy().astype(np.int32)
                if tuple(limb_loc1) not in points:
                    points[tuple(limb_loc1)] = []
                points[tuple(limb_loc1)].append(limb_id)

                limb_loc2 = keypoints[limbs[limb_id][1], :2].detach().numpy().astype(np.int32)
                if tuple(limb_loc2) not in points:
                    points[tuple(limb_loc2)] = []
                points[tuple(limb_loc2)].append(limb_id)

                vectors.append([list(limb_loc1), list(limb_loc2)])
                score1 = all_scores[person_id, limbs[limb_id][0]].numpy()
                score2 = all_scores[person_id, limbs[limb_id][1]].numpy()
                limb_score = min(score1, score2)
                vectors_scores.append([score1, score2])

                if limb_score > keypoint_threshold:
                    color = colors[idx]  # Используем цвет для линий
                    cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), color, 3)

    return img_copy, points, vectors, vectors_scores


# --------------------------------------------------------------------------


def show_vectors(vectors_list, ax, color='b', marker='.', s=30):
    """
    Отображает векторы на заданной оси с помощью узлов и ребер.

    Эта функция принимает список векторов и визуализирует их на
    переданной оси. Векторы отображаются в виде узлов и соединяющих 
    их линий (ребер). 

    Параметры:
    ----------
    vectors_list : list
        Список векторов, где каждый вектор представлен как список
        из двух точек. Каждая точка представляется в виде списка
        или кортежа с координатами (x, y).
 ax : matplotlib.axes.Axes
        Объект оси Matplotlib, на которой будут отображаться векторы.

    color : str, optional
        Цвет узлов и ребер. По умолчанию 'b' (синий).

    marker : str, optional
        Символ, используемый для отображения узлов. По умолчанию '.'.

    s : int, optional Размер узлов. По умолчанию 30.

    Примечания:
    ----------
    - Функция автоматически определяет уникальные узлы из списка
      векторов и строит граф, используя указанные параметры.
    - Параметры `color`, `marker` и `s` могут быть изменены
      для настройки визуализации.

    Пример:
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> vectors = [[(1, 2), (3, 4)], [(3, 4), (5, 6)], [(1, 2), (5, 6)]]
    >>> show_vectors(vectors, ax, color='r', marker='o', s=50)
    >>> plt.show()
    """
    # Извлечение узлов и ребер
    edges = []
    nodes = set()
    for edge in vectors_list:
        nodes.add(tuple(edge[0]))
        nodes.add(tuple(edge[1]))
        
    for edge in vectors_list:
        edges.append((tuple(edge[0]), tuple(edge[1])))

    # Создание графика
    ax.set_aspect('equal')

    # Отрисовка узлов
    for node in nodes:
        ax.scatter(
            node[0], node[1], c=color,
            marker=marker, s=s
            )

    # Отрисовка ребер
    for edge in edges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], c=color, lw=1)
        
        
# --------------------------------------------------------------------------


def nearest_neighbors(A, B):
    """
    Находит ближайших соседей для каждого элемента в наборе A из набора B.

    Параметры:
    A (numpy.ndarray): Массив точек, для которых необходимо найти ближайших соседей.
    Размерность (n, d), где n — количество точек, d — размерность пространства.
    B (numpy.ndarray): Массив точек, среди которых ищутся
    ближайшие соседи. Размерность (m, d), где m — количество точек.

    Возвращает:
    numpy.ndarray: Индексы ближайших соседей из B для каждой точки в A. Размерность (n,).
    """
    tree = KDTree(B)
    distances, indices = tree.query(A)
    return indices
        
        
# --------------------------------------------------------------------------


def compute_transformation(A, B):
    """
    Вычисляет оптимальное аффинное преобразование (вращение, масштаб и сдвиг) 
    для выравнивания набора точек A с набором точек B.

    Параметры:
    A (numpy.ndarray): Массив исходных точек. Размерность (n, d).
    B (numpy.ndarray): Массив целевых точек. Размерность (n, d).

    Возвращает:
    tuple: Кортеж, содержащий:
        - R (numpy.ndarray): Матрица вращения. Размерность (d, d).
        - t (numpy.ndarray): Вектор сдвига. Размерность (d,).
        - scale (float): Масштабный коэффициент.
    """
    # Центрируем точки
    A_centered = A - np.mean(A, axis=0)
    B_centered = B - np.mean(B, axis=0)

    # Вычисляем матрицу ковариации
    H = A_centered.T @ B_centered 
    
    # SVD разложение 
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T  # Вращение

    # Вычисляем масштаб
    scale = np.sum(S) / np.sum(A_centered ** 2)

    # Вычисляем смещение 
    t = np.mean(B, axis=0) - scale * (R @ np.mean(A, axis=0))

    return R, t, scale
        
        
# --------------------------------------------------------------------------


def apply_transformation(A, R, t, scale):
    """
    Применяет аффинное преобразование к набору точек A.

    Параметры:
    A (numpy.ndarray): Массив точек, к которым применяется преобразование. Размерность (n, d).
    R (numpy.ndarray): Матрица вращения. Размерность (d, d).
    t (numpy.ndarray): Вектор сдвига. Размерность (d,).
    scale (float): Масштабный коэффициент.

    Возвращает:
    numpy.ndarray: Преобразованные точки. Размерность (n, d).
    """
    return (A @ R.T) * scale + t
        
        
# --------------------------------------------------------------------------


def icp(A, B, max_iterations=100, tolerance=1e-6):
    """
    Реализует алгоритм итеративного ближайшего соседа (ICP)
    для выравнивания набора точек A с набором точек B.

    Параметры:
    A (numpy.ndarray): Исходный массив точек. Размерность (n, d).
    B (numpy.ndarray): Целевой массив точек. Размерность (m, d).
    max_iterations (int, optional): Максимальное количество итераций. По умолчанию 100.
    tolerance (float, optional): Порог сходимости. По умолчанию 1e-6.

    Возвращает:
    numpy.ndarray: Преобразованный массив точек A, выровненный с B. Размерность (n, d).
    """
    for i in range(max_iterations):
        indices = nearest_neighbors(A, B)
        B_matched = B[indices]

        R, t, scale = compute_transformation(A, B_matched)

        A_transformed = apply_transformation(A, R, t, scale)

        # Проверка сходимости
        mean_error = np.mean(np.linalg.norm(A_transformed - B_matched, axis=1))
        if mean_error < tolerance:
            break

        A = A_transformed  # Обновляем A для следующей итерации

    return A_transformed
        
        
# --------------------------------------------------------------------------


def get_transformed_vectors(points, vectors, transformed_points):
    """
    Получает преобразованные векторы на основе заданных точек и векторов.

    Эта функция принимает словарь точек, список векторов и список преобразованных точек, 
    и возвращает список преобразованных векторов, где каждый вектор представлен в виде 
    списка преобразованных координат.

    Параметры:
    ----------
    points : dict
        Словарь, где ключами являются уникальные идентификаторы точек,
        а значениями - списки индексов, соответствующих вектору.
 vectors : list
        Список векторов, где каждый вектор представлен в виде индексов точек.

    transformed_points : list Список преобразованных точек, где каждая
    точка представлена в виде массива или списка координат (x, y).

    Возвращает:
    ----------
    list
        Список преобразованных векторов, где каждый вектор содержит 
        списки преобразованных координат.

    Примечания:
    ----------
    - Функция округляет координаты преобразованных точек до целых чисел.
    - Если вектор не содержит точек, соответствующих ключам в словаре `points`, 
      он будет представлен как пустой список.

    Пример:
    --------
    >>> points = {'A': [0, 1], 'B': [1, 2]}
    >>> vectors = [[0, 1], [1, 2]]
    >>> transformed_points = [[10.5, 20.5], [30.2, 40.8], [50.1, 60.9]]
    >>> result = get_transformed_vectors(points, vectors, transformed_points)
    >>> print(result)
    [[(10, 20), (30, 40)], [(30, 40), (50, 60)]]
    """
    transformed_vectors = []
    points_keys = list(points.keys())

    for idx in range(len(vectors)):
        vector = []
        for i, key in enumerate(points_keys):
            if idx in points[key]:
                vector.append(list(np.round(transformed_points[i]).astype('int32')))
        transformed_vectors.append(vector)
        
    return transformed_vectors


# --------------------------------------------------------------------------


def get_vectors(keypoints, limbs):
    vectors = []
    points = OrderedDict()

    for limb_id in range(len(limbs)):
        limb_loc1 = keypoints[limbs[limb_id][0], :2].astype(np.int32)
        if tuple(limb_loc1) not in points:
            points[tuple(limb_loc1)] = []
        points[tuple(limb_loc1)].append(limb_id)

        limb_loc2 = keypoints[limbs[limb_id][1], :2].astype(np.int32)
        if tuple(limb_loc2) not in points:
            points[tuple(limb_loc2)] = []
        points[tuple(limb_loc2)].append(limb_id)

        vectors.append([list(limb_loc1), list(limb_loc2)])
        
    return vectors


# --------------------------------------------------------------------------


def cossim(vectors1, vectors2):
    """
    Вычисляет косинусное сходство между парами векторов.

    Эта функция принимает два списка векторов и вычисляет косинусное сходство 
    для каждой пары векторов. Векторы центрируются, чтобы их центры совпадали
    с началом координат перед вычислением сходства.

    Параметры:
    ----------
    vectors1 : list 
        Список векторов (каждый вектор - это массив или список координат),
        для которых будет вычислено косинусное сходство.

    vectors2 : list 
        Список векторов (каждый вектор - это массив или список координат),
        с которыми будет вычислено косинусное сходство. Должен иметь ту же длину,
        что и `vectors1`.

    Возвращает:
    ----------
    tuple
        Кортеж, содержащий:
        - mean_cossim : float
            Среднее значение косинусного сходства для всех пар векторов.
        - similarities : list Список косинусных сходств для каждой пары векторов.
        - shifted_vecs1 : list
            Список векторов из `vectors1`, смещенных в начало координат.
        - shifted_vecs2 : list
            Список векторов из `vectors2`, смещенных в начало координат.

    Примечания:
    ----------
    - Если вектор имеет нулевую норму, косинусное сходство для этой пары будет равно 0.
    - Векторы должны быть представлены в виде массивов или списков, и их размеры должны совпадать.

    Пример:
    --------
    >>> import numpy as np
    >>> vectors_a = [np.array([[1, 2], [2, 3]]), np.array([[4, 5], [5, 6]])]
    >>> vectors_b = [np.array([[2, 3], [3, 4]]), np.array([[5, 6], [6, 7]])]
    >>> mean_similarity, similarities, shifted_a, shifted_b = cossim(vectors_a, vectors_b)
    >>> print(mean_similarity)
    >>> print(similarities)
    """
    def cosine_similarity(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0

    # Основной процесс
    similarities = []
    shifted_vecs1, shifted_vecs2 = [], []
    for vec1, vec2 in zip(vectors1, vectors2):
        # Вычисление центра первой фигуры
        center1 = np.mean(vec1, axis=0)
        
        # Вычисление центра второй фигуры
        center2 = np.mean(vec2, axis=0)
        
        # Перенос векторов в начало координат
        shifted_vec1 = np.array(vec1) - center1
        shifted_vec2 = np.array(vec2) - center2
        
        shifted_vecs1.append(shifted_vec1)
        shifted_vecs2.append(shifted_vec2)
        
        # Преобразуем векторы в одномерные массивы для косинусного сходства
        flat_vec1 = shifted_vec1.flatten()
        flat_vec2 = shifted_vec2.flatten()

        # Вычисление косинусного сходства
        sim = cosine_similarity(flat_vec1, flat_vec2)
        similarities.append(sim)
        
    mean_cossim = np.mean(similarities)
    
    return mean_cossim, similarities, shifted_vecs1, shifted_vecs2


# --------------------------------------------------------------------------


def weight_similarity(pose1, pose2, conf1):
    """
    Вычисляет взвешенное сходство между двумя позами на основе
    их координат и оценок достоверности.

    Эта функция принимает две позы и соответствующие оценки достоверности,
    нормализует позы и вычисляет взвешенное расстояние между ними. 
    В резуьтате возвращается значение сходства в диапазоне от 0 до 1, 
    где 1 означает полное сходство, а 0 - отсутствие сходства.

    Параметры:
    ----------
    pose1 : array-like
        Первая поза, представляемая в виде массива или списка координат. 
        Должна быть одномерной или двумерной структурой.

    pose2 : array-like
        Вторая поза, представляемая в виде массива или списка координат. 
        Должна иметь ту же форму, что и `pose1`.

    conf1 : array-like
        Оценки достоверности для каждой точки в `pose1`, 
        представленные в виде массива или списка. 
        Должны иметь ту же длину, что и `pose1`.

    Возвращает:
    ----------
    float
        Взвешенное сходство между двумя позами, значение в диапазоне от 0 до 1.

    Примечания:
    ----------
    - Если любая из норм поз равна 0 (что означает, что поза не содержит информации), 
      функция возвращает 0.
    - Оценки достоверности предполагается, что каждая точка в позе 
      соответствует одной оценке.

    Пример:
    --------
    >>> import numpy as np >>> pose_a = [[1, 2], [3, 4]]
    >>> pose_b = [[1, 2], [4, 5]]
    >>> confidence = [0.8, 0.9]
    >>> similarity = weight_similarity(pose_a, pose_b, confidence)
    >>> print(similarity)
    """
    flat_pose1 = np.array(pose1).flatten()
    flat_pose2 = np.array(pose2).flatten()
    flat_conf1 = np.array(conf1).flatten()
    
    # Нормализация поз
    norm1 = np.linalg.norm(flat_pose1)
    norm2 = np.linalg.norm(flat_pose2)

    # Если любая из норм равна 0, возвращаем 0
    if norm1 == 0 or norm2 == 0:
        return 0

    # Нормализуем позы
    pose1_normalized = flat_pose1 / norm1
    pose2_normalized = flat_pose2 / norm2
    
    # D(U,V) = (1 / sum(conf1)) * sum(conf1 * ||pose1 - pose2||) = sum1 * sum2
    # Вычисление взвешенного расстояния
    sum1 = 1 / np.sum(flat_conf1)
    sum2 = 0

    for i in range(len(pose1_normalized)):
        # Предполагаем, что каждая точка отвечает одной оценке достоверности
        conf_ind = math.floor(i / 2)  
        sum2 += flat_conf1[conf_ind] * abs(pose1_normalized[i] - pose2_normalized[i])

    weighted_distance = sum1 * sum2

    # Вычисляем взвешенную схожесть (переводим в диапазон от 0 до 1)
    # Сначала проверим, что расстояние не превышает 1
    similarity = max(0, 1 - weighted_distance)

    return similarity


# --------------------------------------------------------------------------


def frames2keypoints(path2modelFrames, path2studentFrames, start_IDX=50):
    """
    Извлекает ключевые точки из кадров изображений,
    используя модель для анализа поз.

    Эта функция принимает пути к двум наборам кадров изображений
    (модель и студента), обрабатывает их с использованием модели 
    ключевых точек, и возвращает список результатов, содержащий 
    изображения с наложенными скелетами и вычисленные сходства.

    Параметры:
    ----------
    path2modelFrames : str 
        Путь к папке с кадрами модели. Кадры должны быть изображениями,
        доступными для обработки.

    path2studentFrames : str 
        Путь к папке с кадрами студента. Кадры должны быть изображениями,
        доступными для обработки.

    start_IDX : int, optional 
        Индекс, с которого начинать обработку кадров. По умолчанию 50.
        Используется для пропуска первых кадров в случае необходимости.

    Возвращает:
    ----------
    list
        Список, содержащий результаты для каждого обработанного кадра. 
        Каждый элемент списка представляет собой другой список, содержащий:
        - result_img_with_skeleton1 : array
            Изображение модели с наложенными ключевыми точками.
        - result_img_with_skeleton2 : array
            Изображение студента с наложенными ключевыми точками.
        - mean_cossim : float
            Среднее значение косинусного сходства между векторами поз.
        - similarities : list
            Список косинусных сходств для каждой пары векторов поз.
        - weighted_similarity : float
            Взвешенное сходство между векторами поз.
        - mean_similarity : float
            Среднее сходство, рассчитанное как среднее взвешенного и косинусного сходства.

    Примечания:
    ----------
    - Функция использует модель ключевых точек (keypointrcnn) для 
      извлечения ключевых точек из изображений.
    - Для обработки изображений используется библиотека OpenCV и PyTorch.
    - Перед использованием функции необходимо убедиться, что модель
      и необходимые функции (например, draw_skeleton_per_person, icp,
      cossim, weight_similarity) корректно загружены и инициализированы.

    Пример:
    --------
    >>> path_model = "/path/to/model/frames"
    >>> path_student = "/path/to/student/frames"
    >>> results = frames2keypoints(path_model, path_student, start_IDX=50)
    >>> print(len(results))  # Количество обработанных кадров 
    """
    # Получаем списки кадров в папках
    frames1_names = os.listdir(path2modelFrames)
    frames2_names = os.listdir(path2studentFrames)
    outputs = []
    # Установка модели в режим оценки
    keypointrcnn.eval()
    # Прогоняем изображения через модель 
    with torch.no_grad():
        # Используем tqdm для отображения прогресса
        for i in tqdm(range(start_IDX, len(frames1_names)), desc="Processing frames"):
            # Формируем пути к изображениям
            frame1_path = os.path.join(path2modelFrames, frames1_names[i])
            frame2_path = os.path.join(path2studentFrames, frames2_names[i])
            # Загружаем изображения
            image1 = cv2.cvtColor(cv2.imread(frame1_path), cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(cv2.imread(frame2_path), cv2.COLOR_BGR2RGB)
            # Прменение трансформации к изображениям
            tensor_image1 = transform(image1)
            tensor_image2 = transform(image2)
            # Создание батча из двух изображений 
            batch_tensor = torch.stack((tensor_image1, tensor_image2), dim=0)
            # Пропускаем изображения через модель
            output = keypointrcnn(batch_tensor)
            
            # Извлекаем ключевые точки и уверенности
            keypoints1 = output[0]['keypoints']
            scores1 = output[0]['keypoints_scores']
            boxes1 = output[0]['boxes']
            confs1 = output[0]['scores']

            keypoints2 = output[1]['keypoints']
            scores2 = output[1]['keypoints_scores']
            boxes2 = output[1]['boxes']
            confs2 = output[1]['scores']
            
            # Получаем изображения и данные для работы с векторами
            # [:1] — ограничим работу первыми объектами
            [result_img_with_skeleton1,
             points1, vectors1, vec_scores1] = draw_skeleton_per_person(
                image1, keypoints1[:1], scores1[:1], confs1[:1], limbs, boxes1[:1],
                keypoint_threshold=2, conf_threshold=0.9
            )
            [result_img_with_skeleton2,
             points2, vectors2, vec_scores2] = draw_skeleton_per_person(
                image2, keypoints2[:1], scores2[:1], confs2[:1], limbs, boxes2[:1],
                keypoint_threshold=2, conf_threshold=0.9
            )
            
            # Преобразуем ключи OrderedDict в список уникальных точек
            unique_points1 = list(points1.keys())
            unique_points2 = list(points2.keys())

            # Соберём два набора ключевых точек
            Y = np.array(list(unique_points1)) # [[x1,y1],[x2,y2],...]
            X = np.array(list(unique_points2)) # [[x1,y1],[x2,y2],...]
            
            transformed_X = icp(X, Y)
            transformed_vectors2 = get_transformed_vectors(
                points2, vectors2, transformed_X
                )
            
            # Вычисление косинусного сходства
            mean_cossim, similarities, shifted_vecs1, shifted_vecs2 = cossim(
                vectors1, transformed_vectors2
                )
            # Вычисление взвешенного сходства
            weighted_similarity = weight_similarity(
                shifted_vecs1, shifted_vecs2, vec_scores1
                )
            # Среднее сходство
            mean_similarity = (weighted_similarity + mean_cossim)/2
            # Складываем полученное в список
            outputs.append([
                result_img_with_skeleton1, result_img_with_skeleton2,
                mean_cossim, similarities, weighted_similarity, mean_similarity,
                keypoints1, keypoints2
                ])
            
    return outputs


# --------------------------------------------------------------------------


def outputs2gif(relevant_frame_names, outputs, path2folder, gif_filename='dancing.gif'):
    """
    Создает GIF-анимацию из списка изображений, содержащих результаты
    обработки кадров с наложенными скелетами.

    Функция принимает список выходных данных, который включает изображения
    с наложенными ключевыми точками, и создает GIF-файл, в который
    добавляются изображения с метриками сходства. 

    Параметры:
    ----------
    outputs : list
        Список, содержащий результаты обработки кадров. Каждый элемент
        должен быть списком, где первые два элемента — это изображения
        с наложенными скелетами (результаты модели).
    gif_filename : str, optional 
        Имя выходного GIF-файла. По умолчанию 'dancing.gif'.
        Указывает, как будет называться созданный GIF.

    Примечания:
    ----------
    - Для работы функции необходимо, чтобы переменные 
      `relevant_frame_names`, `path2folder`, `mean_cossims`,
      `weighted_sims` и `mean_sims` были доступны в области видимости функции.
    - Функция использует библиотеку PIL для обработки изображений
      и Matplotlib для отображения метрик и создания GIF.
    - GIF будет содержать изображения, состоящие из оригинальных 
      кадров и наложенных скелетов, а также метрики, которые отображаются
      в заголовке каждого кадра.

    Пример:
    --------
    >>> outputs = [...]  # Список выходных данных, полученных из функции frames2keypoints
    >>> outputs2gif(outputs, gif_filename='my_animation.gif')
    >>> # GIF будет сохранен с именем 'my_animation.gif'
    """
    
    mean_cossims = []
    cossims_lists = []
    weighted_sims = []
    mean_sims = []

    for i in range(len(outputs)):
        mean_cossims.append(outputs[i][2])
        cossims_lists.append(outputs[i][3])
        weighted_sims.append(outputs[i][4])
        mean_sims.append(outputs[i][5])
        
    images = []

    for i in tqdm(range(0, len(outputs), 3), desc="Processing frames"):
        result_img_with_skeleton1, result_img_with_skeleton2 = outputs[i][:2]

        full_frame_name = relevant_frame_names[i]
        path2fullFrame = os.path.join(path2folder, full_frame_name)
        fullFrame = plt.imread(path2fullFrame)

        # Изменяем размер изображений
        target_size = (fullFrame.shape[1], fullFrame.shape[0])  # Целевой размер - размер fullFrame 
        result_img_with_skeleton1 = Image.fromarray(
            result_img_with_skeleton1).resize(target_size, Image.LANCZOS)
        result_img_with_skeleton2 = Image.fromarray(
            result_img_with_skeleton2).resize(target_size, Image.LANCZOS)

        # Объединяем второе и третье изображения
        combined_width = result_img_with_skeleton2.width + result_img_with_skeleton1.width 
        combined_height = max(
            result_img_with_skeleton2.height, result_img_with_skeleton1.height
            )

        # Создаем новое изображение для объединения
        combined_image = Image.new('RGB', (combined_width, combined_height))

        # Вставляем второе изображение 
        combined_image.paste(result_img_with_skeleton2, (0, 0))
        # Вставляем третье изображение 
        combined_image.paste(
            result_img_with_skeleton1,
            (result_img_with_skeleton2.width, 0)
            )

        # Рисуем линию склейки
        draw = ImageDraw.Draw(combined_image)
        # Позиция линии склейки
        line_position = result_img_with_skeleton2.width  
        # Рисуем вертикальную линию
        draw.line(
            [(line_position, 0), (line_position, combined_height)],
            fill='white', width=5
            )  

        # Получаем метрики
        mean_cossim = mean_cossims[i]
        weighted_sim = weighted_sims[i]
        mean_sim = mean_sims[i]

        # Создаем фигуру для отображения 
        fig, ax = plt.subplots(1, 2, figsize=(8, 7))

        # Отображаем первое изображение и объединенное изображение 
        ax[0].imshow(fullFrame, aspect='auto')
        ax[1].imshow(combined_image, aspect='auto')

        # Скрываем оси 
        ax[0].axis('off')
        ax[1].axis('off')

        # Установка заголовка с метриками 
        metrics_text = f"""Frame Index: {i}
    Mean cossims: {mean_cossim:.4f}
    Weighted sims: {weighted_sim:.4f}
    Mean sims: {mean_sim:.4f}
    """

        fig.suptitle(metrics_text, fontsize=10)
        plt.subplots_adjust(wspace=0.02)

        # Сохраняем изображение в BytesIO 
        img_byte_arr = BytesIO()
        # Сохраняем в формате PNG
        plt.savefig(img_byte_arr, format='png', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)  # Закрываем фигуру
        img_byte_arr.seek(0)  # Перемещаем указатель в начало

        # Открываем изображение из BytesIO и добавляем его в список
        images.append(Image.open(img_byte_arr))

    # Создаем GIF из списка изображений
    images[0].save(gif_filename, save_all=True, append_images=images[1:], loop=0, duration=3)

    print(f'GIF saved as {gif_filename}')
    
    
# --------------------------------------------------------------------------


def outputs2gif2(
    st_frames_folder, tch_frames_folder, outputs,
    st_files, tch_files, 
    gif_filename='dancing.gif', scale_factor=1.2, figsize=(7, 5)
    ):
    mean_cossims = []
    cossims_lists = []
    weighted_sims = []
    mean_sims = []
    kpts1, kpts2 = [], []
    transformed_vectors2 = []

    for i in range(len(outputs)):
        mean_cossims.append(outputs[i][2])
        cossims_lists.append(outputs[i][3])
        weighted_sims.append(outputs[i][4])
        mean_sims.append(outputs[i][5])
        kpts1.append(outputs[i][6])
        kpts2.append(outputs[i][7])
        transformed_vectors2.append(outputs[i][8])
        
    images = []

    for i in tqdm(range(0, len(outputs), 3), desc="Processing frames"):
        # Получаем метрики
        mean_cossim = mean_cossims[i]
        weighted_sim = weighted_sims[i]
        mean_sim = mean_sims[i]
        vectors1 = get_vectors(kpts2[i], limbs)
        tf_vectors2 = transformed_vectors2[i]
        

        # Создаем фигуру с 4 подграфиками 
        fig, axs = plt.subplots(1, 4, figsize=figsize)  # Увеличиваем размер фигуры 
        # Визуализация левого полукадра 
        left_part_path = os.path.join(st_frames_folder, st_files[i])
        left_half_image = plt.imread(left_part_path)
        axs[0].imshow(left_half_image)
        axs[0].axis('off')

        # Визуализация векторов для левого танцора
        show_vectors(np.array(tf_vectors2) * scale_factor, ax=axs[1])
        axs[1].axis('off')
        axs[1].invert_yaxis()

        # Визуализация векторов для правого танцора
        show_vectors(np.array(vectors1) * scale_factor, ax=axs[2], color='r')
        axs[2].axis('off')
        axs[2].invert_yaxis()

        # Визуализация правого полукадра 
        right_part_path = os.path.join(tch_frames_folder, tch_files[i])
        right_half_image = plt.imread(right_part_path)
        axs[3].imshow(right_half_image)
        axs[3].axis('off')

        # Определяем пределы осей для векторов 
        all_vectors = np.concatenate([tf_vectors2, vectors1], axis=0)
        # Увеличиваем пределы осей для учета масштабирования
        x_min, x_max = np.min(all_vectors[:, 0]), np.max(all_vectors[:, 0])
        y_min, y_max = np.min(all_vectors[:, 1]) - 10, np.max(all_vectors[:, 1]) + 70

        # Устанавливаем одинаковые пределы осей для подграфиков с векторами 
        axs[1].set_xlim(x_min, x_max)
        axs[1].set_ylim(y_max, y_min)  # Инвертируем Y-оси
        axs[2].set_xlim(x_min, x_max)
        axs[2].set_ylim(y_max, y_min)  # Инвертируем Y-оси 

        # Добавление текста с ID кадра и метрикой косинусной схожести
        plt.figtext(0.5, 0.95, f'Frame ID: {i}', ha='center', fontsize=12)
        plt.figtext(0.5, 0.9, f'CosSim: {mean_cossim:.4f}', ha='center', fontsize=12)
        plt.figtext(0.5, 0.85, f'WeightSim: {weighted_sim:.4f}', ha='center', fontsize=12)
        plt.figtext(0.5, 0.85, f'MeanSim: {mean_sim:.4f}', ha='center', fontsize=12)

        plt.subplots_adjust(wspace=0.05)
        plt.tight_layout()

        # Сохраняем изображение в BytesIO 
        img_byte_arr = BytesIO()
        # Сохраняем в формате PNG
        plt.savefig(img_byte_arr, format='png', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)  # Закрываем фигуру
        img_byte_arr.seek(0)  # Перемещаем указатель в начало

        # Открываем изображение из BytesIO и добавляем его в список
        images.append(Image.open(img_byte_arr))

    # Создаем GIF из списка изображений
    images[0].save(gif_filename, save_all=True, append_images=images[1:], loop=0, duration=3)

    print(f'GIF saved as {gif_filename}')


# --------------------------------------------------------------------------


