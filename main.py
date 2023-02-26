import cv2
import cvzone
import pickle
import numpy as np
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# чтение параметров из config.ini
use_default = True
params = 'DEFAULT' if use_default else 'MODIFIED' # использование DEFAULT или MODIFIED (для экспериментов) параметров

WIDTH = int(config[params]['width'])
HEIGHT = int(config[params]['height'])
trsh_block_size = int(config[params]['adaptive_thresholds_block_size'])
C = int(config[params]['adaptive_thresholds_constant'])
limit_board = int(config[params]['white_pixel_limit'])

IMAGE_NAME = "car_park"

with open('positions', 'rb') as f:
    POSITIONS_LIST = pickle.load(f)

cap = cv2.VideoCapture('carPark.mp4')


def croping(img: np.array, row: pd.Series) -> np.array:
    """
    Вырезает из общего кадра изображение каждого парковочного места
    :param img: Общий кадр
    :param row: Координаты и параметры места
    :return: вырезанное изображение в формате numpy array
    """
    return img[row.y:row.y_plus_height, row.x:row.x_plus_width]


def draw_rec(img, row):
    """
    Отрсовка прямоугольников по координатам парковочных мест
    :param img: Общий кард
    :param row: Координаты и параметры места
    :return: None
    """
    cv2.rectangle(img, (row.x, row.y), (row.x_plus_width, row.y_plus_height), row.vacant_color, row.thickness)


while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) #перезапускает цикл после окончания видео

    success, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_post = cv2.adaptiveThreshold(img_gray, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     trsh_block_size if trsh_block_size % 2 else trsh_block_size + 1,
                                     C)

    df = pd.DataFrame(POSITIONS_LIST, columns=(['x', 'y']))
    df['x_plus_width'] = df['x'] + WIDTH
    df['y_plus_height'] = df['y'] + HEIGHT #достроение полных координат
    df['img_crop'] = df.apply(lambda row: croping(img_post, row), axis=1) #изображение парковочного места
    df['white_pix_count'] = df['img_crop'].apply(lambda x: cv2.countNonZero(x))  # детекция автомобиля
    df['vacant_color'] = df['white_pix_count'].apply(lambda x: (0, 255, 0) if x < limit_board else (0, 0, 255))  #цвет рамки
    df['thickness'] = np.where(df['white_pix_count'] < limit_board, 4, 2) #толщина рамки
    df = df.sort_values('vacant_color', ascending=True)

    df.apply(lambda row: draw_rec(img, row), axis=1)
    cvzone.putTextRect(img,
                       f'Vacant: {(df["white_pix_count"] < limit_board).sum()}/{len(POSITIONS_LIST)}',
                       (100, 50),
                       scale=2,
                       thickness=2,
                       offset=20,
                       colorR=(100, 100, 100))
    cv2.imshow(IMAGE_NAME, img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
