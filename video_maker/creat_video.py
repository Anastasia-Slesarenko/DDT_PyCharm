import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import multiprocessing as mp


n_jobs = 3
summ_file = 'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/test_frames_drops' \
            '/cap_cut_crop_drops.mp4/sum_df_fix_new.cvs'
test_im = 'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/test_frames_drops' \
            '/cap_cut_crop_drops.mp4/0000000000.jpg'
path_frames = 'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/test_frames_drops' \
            '/cap_cut_crop_drops.mp4/'
save_videoTo = 'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/test_frames_drops' \
            '/video_maker/'

sum_df = pd.read_csv(summ_file, index_col=0)
img = Image.open(test_im)
w, h = img.size
# тут все фреймы
t_range = np.arange(65120, sum_df['t_sum'].max())
n_frame = int(t_range.max())
# тут только фреймы с разрядами
t_r = sum_df['t_sum'].unique()


def pros_frame(number):
    frame_name = '{0:0>10}.jpg'.format(number)
    # frame_with_name = Image.open(f'{path_frames}{frame_name}')
    img = cv2.imread(path_frames + frame_name)

    if number in t_r:
        plot_df = sum_df[sum_df.t_sum == number]
        img[plot_df.y_sum.astype(int).tolist(), plot_df.x_sum.astype(int).tolist(), -1] = 255
        img[plot_df.y_sum.astype(int).tolist(), plot_df.x_sum.astype(int).tolist(), :-1] = 0
    return (number, img)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ==================================================================
    # asinc multi process image + save
    # progress bar
    pbar = tqdm(total=len(t_range))
    # list with pross frames
    frame_array = []
    # update function
    def update(out):
        global frame_array, pbar
        frame_array += [out]
        # print(out[1])
        pbar.update()

    pool = mp.Pool(n_jobs)  # open pools
    for name_frame in t_range:
        pool.apply_async(pros_frame, args=(name_frame, ), callback=update)

    pool.close()
    pool.join()
    pbar.close()

    # ==================================================================
    # make video
    # sort frames by number
    frame_array = sorted(frame_array, key=lambda tup: tup[0])
    # get size
    height, width, layers = frame_array[0][1].shape
    size = (width, height)

    out = cv2.VideoWriter(save_videoTo + 'vid.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps=25,
                          frameSize=size)
    for frame in frame_array:
        out.write(frame[1])
    out.release()