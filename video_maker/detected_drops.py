import cv2
import numpy as np
from moviepy.editor import VideoFileClip

if __name__ == '__main__':

    def find_cont(imag):
        blu = cv2.GaussianBlur(imag, (3,3), 0)
        #T, th = cv2.threshold(blu, 215, 255, cv2.THRESH_BINARY)
        (cnts, _) = cv2.findContours(blu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return cnts

    cap = cv2.VideoCapture(
        r'C:\Users\1\Desktop\VKR\Master\data\silicone SPBU\PKD_02.02.22_part\2\cap_cut_crop_drops.mp4')
    fgbg = cv2.createBackgroundSubtractorMOG2()

    cap2 = cv2.VideoCapture(
        r'C:\Users\1\Desktop\VKR\Master\data\silicone SPBU\PKD_02.02.22_part\2\help_video.mp4')

    cap2.set(1, 30)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r1 = np.array(frame)

    cap2.set(1, 15)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r2 = np.array(frame)
    r = np.array([r1, r2])
    r = np.mean(r,0)
    print(r.shape)
    cv2.imshow('frame_d', r)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray) - r
        #gray = fgbg.apply(gray)
        #fr = find_cont(gray)
        #cv2.drawContours(gray, fr, -1, (255,0,0), 3, cv2.LINE_AA)

        cv2.imshow('fgmask', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# def get_cut_and_crop_cap(path_cap, path_file_coord, time_start, time_end, name_video):
#     """
#     :param name_video:
#     :param path_cap: путь к видео
#     :param path_file_coord: путь к файлу с координатами образца с искрами
#     :param time_start: время начала испытания
#     :param time_end: время конца испытания
#     :return:
#     """
#
#     # Загрузка видео для нахождения высоты и ширины
#     cap_cut = VideoFileClip(path_cap)
#     h = cap_cut.h
#     w = cap_cut.w
#
#     # Считываем координаты образца с искарми из файла "coord_p.txt" для обрезки видео
#     f = open(path_file_coord, mode='r', encoding='utf8', newline='\r\n')
#     coord_p = []
#     for i, line in enumerate(f):
#         coord_p += [list(map(int, line.split(',')))]
#     y_start = coord_p[0][1]
#     x_start = coord_p[2][0]
#     y_end = coord_p[1][1]
#     x_end = coord_p[0][0]
#
#     # Обрезаем видео по длительности и вырезаем образец для обработки искр
#     # Сохранем новое видео как cap_cut_crop
#     cap_cut_new = cap_cut \
#         .subclip(time_start, time_end).crop(x1=0, y1=0, x2=w // 2, y2=h).crop(x1=x_start, y1=y_start, x2=x_end,
#                                                                               y2=y_end)
#     cap_cut_new.write_videofile(name_video)
#     cap_cut_new.close()
#
# if __name__ == '__main__':
#     path_new = 'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/'
#
#     get_cut_and_crop_cap(path_cap=f'{path_new}2022-02-02 13-02-41.mp4',
#                          path_file_coord=f"{path_new}coord_d.txt", time_start=0,
#                          time_end=VideoFileClip(f'{path_new}2022-02-02 13-02-41.mp4').duration,
#                          name_video='C:/Users/1/Desktop/VKR/Master/data/'
#                                     '/silicone SPBU/PKD_02.02.22_part/2/help_video.mp4')