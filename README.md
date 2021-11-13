**Тест на стекание капель**
## Описание проекта
Проект состоит из нескольких файлов -py. Они позволят загрузить видео, 
сделать раскадровку и получить координаты разрядов на образце. Постобработка 
позволяет построить карту разрядов, зависимость суммарной яркости от времени и т.д.
## Содержание проекта
1. `get_coordinates.py` - возвращает координаты углов образца на камере с разрядами
и на камере с каплями в соответствующих файлах: 'coord_p.txt', 'coord_d.txt'. На входе необходимо
указать путь к видео и время в секундах, когда образец хорошо видно на двух камерах. На всплывающих изображениях указывать 
мышкой углы, начиная с правого верхнего по часовой.
2. `get_cut_and_crop_cap.py` - возвращает видео с камеры для разрядов, вырезанное по координатам образца: 'cap_cut_crop.mp4'.
На входе необходимо указать путь к видео, путь к файлу 'coord_p.txt', время начала и конца испытания.
3. `get_frames_of_discharges_from_cap.py` - возвращает раскадрованное видео 'cap_cut_crop.mp4' в папку test_frames.
На входе необхомо указать путь к видео 'cap_cut_crop.mp4'.
4. `get_xy_of_pixeles.py` - возвращает txt файл с координатами ярких пикселей для каждого
кадра, на которм был найден разряд. Название файла соответствует номеру кадра. На входе функции 'get_frames_with_discharges'
необходимо указать путь к папке с кадрами, и порого по количеству ярких пикселей на кадре (10 по умолчанию). Дополнительно указать путь 
к первому кадру frames_0 для функции 'find_false_pixels'.
5. `let's_try_again.py` - отрисовывает графики 'Зависимость разрядов вдоль y-координаты от времени',
'Распределение яркости разрядов по длине образца', 'Зависимость яркости разрядов от времени'. Необходимо
указать путь к папке с txt файлами координат разрядов.
6. `get_figure.py` - отрисовывает карту разрядов. С помощью ползунка возможно регулировать количество
кадров и наблюдать развитие во времени. Необходимо указать путь к файлу 'sum_df.cvs'. Если файла нет,
возможно загрузить его из `let's_try_again.py` (часть закомментированного кода).

Загрузить исходное видео теста можно по ссылке: https://drive.google.com/file/d/10HaFx7Lss_BbnKsAgVfW6lz6_XVsAsjg/view?usp=sharing


