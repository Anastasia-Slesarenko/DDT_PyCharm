from moviepy.editor import VideoFileClip


def get_cut_and_crop_cap(path_cap, path_file_coord, time_start, time_end):
    """
    :param path_cap: путь к видео
    :param path_file_coord: путь к файлу с координатами образца с искрами
    :param time_start: время начала испытания
    :param time_end: время конца испытания
    :return:
    """

    # Загрузка видео для нахождения высоты и ширины
    cap_cut = VideoFileClip(path_cap)
    h = cap_cut.h
    w = cap_cut.w

    # Считываем координаты образца с искарми из файла "coord_p.txt" для обрезки видео
    f = open(path_file_coord, mode='r', encoding='utf8', newline='\r\n')
    coord_p = []
    for i, line in enumerate(f):
        coord_p += [list(map(int, line.split(',')))]
    y_start = coord_p[0][1]
    x_start = coord_p[2][0]
    y_end = coord_p[1][1]
    x_end = coord_p[1][0]

    # Обрезаем видео по длительности и вырезаем образец для обработки искр
    # Сохранем новое видео как cap_cut_crop
    cap_cut_new = cap_cut \
        .subclip(time_start, time_end).crop(x1=w // 2, y1=0, x2=w, y2=h).crop(x1=x_start, y1=y_start, x2=x_end,
                                                                              y2=y_end)
    cap_cut_new.write_videofile('cap_cut_crop.mp4')
    cap_cut_new.close()


if __name__ == '__main__':

    get_cut_and_crop_cap(path_cap=
                         'C:/Users/1/Desktop/ВКР/Магистерская/ProjectPyCharm/DDT_PyCharm/2021-10-22 16-12-08.mp4',
                         path_file_coord="coord_p.txt", time_start=128*60, time_end=168*60)
