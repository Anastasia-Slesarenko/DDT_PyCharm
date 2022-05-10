from moviepy.editor import VideoFileClip
from tqdm import tqdm

def get_cut_and_crop_cap(path_cap, path_file_coord, time_start, time_end, name_video):
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
    x_end = coord_p[0][0]

    # Обрезаем видео по длительности и вырезаем образец для обработки искр
    # Сохранем новое видео как cap_cut_crop
    cap_cut_new = cap_cut \
        .subclip(time_start, time_end).crop(x1=w // 2, y1=0, x2=w, y2=h).crop(x1=x_start, y1=y_start, x2=x_end,
                                                                              y2=y_end)
    cap_cut_new.write_videofile(name_video)
    cap_cut_new.close()


if __name__ == '__main__':

    # n = ['2022-02-02 11-45-37', '2022-02-02 11-49-18', '2022-02-02 11-50-35', '2022-02-02 11-51-46',
    #      '2022-02-02 11-53-42', '2022-02-02 11-55-39']
    # path_new = 'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/CD/'
    #
    # i_k = [1, 2, 3, 4, 5, 6]
    #
    # for k, i in tqdm(zip(n, i_k)):
    #     get_cut_and_crop_cap(path_cap=f'{path_new}{k}.mp4',
    #                          path_file_coord=f"{path_new}coord_p.txt", time_start=0,
    #                          time_end=VideoFileClip(f'{path_new}{k}.mp4').duration,
    #                          name_video=f'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/CD/cap_cut_{i}.mp4')

    n = ['2022-05-05 10-56-31', '2022-05-05 14-01-50']
    k = [1, 2]
    path_new = 'C:/Users/1/Desktop/VKR/Master/data/new silicone/'
    for i, m in zip(k, n):
        get_cut_and_crop_cap(path_cap=f'{path_new}{m}.mp4',
                             path_file_coord=f"{path_new}coord_p.txt", time_start=0,
                             time_end=VideoFileClip(f'{path_new}{m}.mp4').duration,
                             name_video=f'C:/Users/1/Desktop/VKR/Master/data/new silicone/cap_cut_{i}.mp4')


