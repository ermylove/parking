import cv2
import pickle
import configparser

try:
    with open('positions', 'rb') as f:
        pos_list = pickle.load(f)
except:
    pos_list = []

config = configparser.ConfigParser()
config.read("config.ini")

WIDTH = int(config['DEFAULT']['width'])
HEIGHT = int(config['DEFAULT']['height'])


def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        pos_list.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(pos_list):
            x1, y1 = pos
            if x1 < x < x1 + WIDTH and y1 < y < y1 + HEIGHT:
                pos_list.pop(i)

    #with open('positions', 'wb') as f:
    #   pickle.dump(pos_list, f)


while True:
    img = cv2.imread('carParkImg.png')
    for pos in pos_list:
        cv2.rectangle(img, pos, (pos[0] + WIDTH, pos[1] + HEIGHT), (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
