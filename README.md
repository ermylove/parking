# parking

## Задачи: 
Определение в видеозаписи заполненость парковочных мест и подсчет количества свободных мест.

### labeling.py
Скрипт для разметки парковочных мест
![image](https://user-images.githubusercontent.com/97171534/221420918-d7087f8d-43ce-431c-acf8-c903f90f0ae0.png)

### setting.py
Подбор параметров для преобразования изображения.
Параметры сохранены в config['DEFAULT'] эксперементальные параметры будут обновлять config['MODIFIED']
![image](https://user-images.githubusercontent.com/97171534/221421167-4335840b-c943-4378-97ac-d56c74b05f77.png)
![image](https://user-images.githubusercontent.com/97171534/221421180-ceb0397f-2fa1-4b9f-b288-aeb8c597c935.png)


### main.py
Детекция заполненость парковочных мест и подсчет количества свободных мест на примере вложенного видео carPark.mp4
![image](https://user-images.githubusercontent.com/97171534/221421285-a17e76e2-821c-469d-a95d-ba1b8ed8db66.png)
