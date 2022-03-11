import cv2
from os import path, remove


# Поиск координатов пикселя при нажатии левой кнопки мыши
def click_event(event, x, y, flags, params):
    # По часовой стрелки, начиная с верхнего правого угла
    if event == cv2.EVENT_LBUTTONDOWN:
        with open(params[0], 'a') as out:
            out.write(f'{x},{y}' + '\n')
        # Вывод координат на изображении
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(params[1], str(x) + ',' +
                    str(y), (x, y + 30), font,
                    0.8, (0, 0, 255), 2)
        cv2.circle(params[1], (x, y), radius=0, color=(0, 0, 255), thickness=4)
        cv2.imshow(params[2], params[1])


def get_coordinates(path_cap, time_find_sample):
    """
    :param path_cap: указываем путь видео
    :param time_find_sample: задаем время в секундах, когда образец хорошо видно на двух камерах
    :return:
    """
    # Загрузка видео
    cap = cv2.VideoCapture(path_cap)
    # Поиск ширины w, высоты h, частоты fps и количество кадров n_frames в видео
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Поиск нужного кадра для выделения контуров
    number = fps * time_find_sample
    total_frames = cap.get(7)
    cap.set(1, number)
    ret, frame = cap.read()

    # Выделяем кадр frame_d, где видны капли
    frame_d = frame[0:h, 0:w // 2]
    cv2.imshow('frame_d', frame_d)
    # Указываем мышкой углы образца по часовой стрелке, начиная с правого верхнего угла
    # Получаем координаты углов образца с каплями в 'coord_d.txt'
    if path.isfile(f'{path_cord}coord_d.txt'):
        remove(f'{path_cord}coord_d.txt')
    cv2.setMouseCallback('frame_d', click_event, [f'{path_cord}coord_d.txt', frame_d, 'coordinates on frame of drops'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Выделяем кадр frame_p, где видны искры
    frame_p = frame[0:h, w // 2:w]
    cv2.imshow('frame_p', frame_p)
    # Указываем мышкой углы образца по часовой стрелке, начиная с правого верхнего угла
    # Получаем координаты углов образца с искрами в 'coord_p.txt'
    if path.isfile(f'{path_cord}coord_p.txt'):
        remove(f'{path_cord}coord_p.txt')
    cv2.setMouseCallback('frame_p', click_event, [f'{path_cord}coord_p.txt', frame_p, 'coordinates on frame of discharges'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path_cord = '/silicone SPBU/PKD_18.02.22_part/10/'
    get_coordinates(path_cap='/silicone SPBU/PKD_18.02.22_part/10/2022-02-18 15-28-35.mp4',
                    time_find_sample=0)


