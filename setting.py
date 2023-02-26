import cv2
import cvzone
import pickle
import numpy as np
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# чтение параметров из config.ini в DEFAULT
WIDTH = int(config['DEFAULT']['width'])
HEIGHT = int(config['DEFAULT']['height'])
block_size = int(config['DEFAULT']['adaptive_thresholds_block_size'])
C = int(config['DEFAULT']['adaptive_thresholds_constant'])
limit_board = int(config['DEFAULT']['white_pixel_limit'])

IMAGE_NAME = "Image"

with open('positions', 'rb') as f:
    POSITIONS_LIST = pickle.load(f)

cv2.namedWindow(IMAGE_NAME)
cv2.createTrackbar("block Size", IMAGE_NAME, block_size, 50, lambda some: None)
cv2.createTrackbar("Constant", IMAGE_NAME, C, 50, lambda some: None)
cv2.createTrackbar("Limit", IMAGE_NAME, limit_board, 5000, lambda some: None)


def croping(img: np.array, row: pd.Series) -> np.array:
    """
    Вырезает из общего кадра изображение каждого парковочного места
    :param img: Общий кадр
    :param row: Координаты и параметры места
    :return: вырезанное изображение в формате numpy array
    """
    return img[row.y:row.y_plus_height, row.x:row.x_plus_width]


def draw_rec(img: np.array, row: pd.Series) -> None:
    """
    Отрсовка прямоугольников по координатам парковочных мест
    :param img: Общий кард
    :param row: Координаты и параметры места
    :return: None
    """
    cv2.rectangle(img, (row.x, row.y), (row.x_plus_width, row.y_plus_height), row.vacant_color, row.thickness)


def past_score(img: np.array, row: pd.Series) -> None:
    """
    Отрисовка значения белых пикселей для каждого парковочного места
    :param img: Общий кард
    :param row: Координаты и параметры места
    :return: None
    """
    cvzone.putTextRect(img, str(row.white_pix_count), (row.x, row.y_plus_height), scale=1,
                       thickness=2, offset=0, colorR=(0, 0, 0))


def plt2arr(series: pd.Series) -> np.array:
    """
    Преобразование графика в numpy array
    :param series: Параметры для построения графика
    :return: numpy array
    """
    fig = (series // 100).hist(bins=30, grid=False, histtype='step').get_figure()
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w, h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    fig.clear()
    return rgba_arr


while True:
    trsh_block_size = cv2.getTrackbarPos("block Size", IMAGE_NAME)
    C = cv2.getTrackbarPos("Constant", IMAGE_NAME)
    limit_board = cv2.getTrackbarPos("Limit", IMAGE_NAME)

    img = cv2.imread('carParkImg.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_post = cv2.adaptiveThreshold(img_gray, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     trsh_block_size if trsh_block_size % 2 else trsh_block_size + 1,
                                     C)

    df = pd.DataFrame(POSITIONS_LIST, columns=(['x', 'y']))
    df['x_plus_width'] = df['x'] + WIDTH
    df['y_plus_height'] = df['y'] + HEIGHT
    df['img_crop'] = df.apply(lambda row: croping(img_post, row), axis=1)
    df['white_pix_count'] = df['img_crop'].apply(lambda x: cv2.countNonZero(x))
    df['vacant_color'] = df['white_pix_count'].apply(lambda x: (0, 255, 0) if x < limit_board else (0, 0, 255))
    df['thickness'] = np.where(df['white_pix_count'] < limit_board, 2, 6)
    df = df.sort_values('vacant_color', ascending=True)

    hist_img = cv2.cvtColor(plt2arr(df['white_pix_count']), cv2.COLOR_BGR2GRAY)

    img = cv2.cvtColor(img_post, cv2.COLOR_GRAY2RGB)
    df.apply(lambda row: draw_rec(img, row), axis=1)
    df.apply(lambda row: past_score(img, row), axis=1)
    cv2.imshow(IMAGE_NAME, img)
    cv2.imshow('white density', hist_img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        # запись новых параметров в config.ini в MODIFIED
        config['MODIFIED']['adaptive_thresholds_block_size'] = str(trsh_block_size)
        config['MODIFIED']['adaptive_thresholds_constant'] = str(C)
        config['MODIFIED']['white_pixel_limit'] = str(limit_board)

        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        break
