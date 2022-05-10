from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video
import matplotlib as mpt
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter


if __name__ == '__main__':

    # mpt.rcParams['animation.ffmpeg_path'] = r'C:\\Users\1\Desktop\VKR\Master\ProjectPyCharm\DDT_PyCharm\venv\Lib\site-packages\ffmpeg'

    sum_df = pd.read_csv('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/test_frames_drops'
                         '/cap_cut_crop_drops.mp4/sum_df_fix.cvs', index_col=0)
    img = Image.open('C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/test_frames_drops'
                     '/cap_cut_crop_drops.mp4/0000000000.jpg')
    w, h = img.size

    # тут все фреймы
    t_range = np.arange(1, sum_df['t_sum'].max())
    n_frame = int(t_range.max())

    # тут только фреймы с разрядами
    t_r = sum_df['t_sum'].unique()


    def au_frame(number, ax):

        frame_name = '{0:0>10}.jpg'.format(number)
        path_frames = 'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/' \
                      'test_frames_drops/cap_cut_crop_drops.mp4/'
        # frame_with_name = Image.open(f'{path_frames}{frame_name}')
        frame_with_name = plt.imread(f'{path_frames}{frame_name}')

        ax.imshow(frame_with_name)

        if number in t_r:
            plot_df = sum_df[sum_df.t_sum == number]
            plot_df.plot(x='x_sum', y='y_sum', ax=ax, c='r', kind='scatter')
            ax.set_xlabel('')
            ax.set_ylabel('')

        ax.grid(ls='--')


    fig, ax = plt.subplots()

    # Make a camera of the figure
    camera = Camera(fig)

    # Animation = lambda i: au_frame(i, ax)
    # anim = FuncAnimation(fig, Animation, frames=200) # len(t_range))
    # writer = PillowWriter(fps=25)
    # anim.save('video_maker/test_video2.mp4', writer=writer)

    for number in tqdm(t_range[-200:]):
        # Our animation will have each frame be a different value of k from the list above

        # For a given value of k, we'll plot our function with that k-value
        camera.snap()

    # Make the animation
    animation = camera.animate()
    # Stop the empty plot from displaying
    plt.close()
    # Save the animation-- notes below
    animation.save('video_maker/test_video2.mp4', fps=25)
    # Show the video you've just saved
    Video("video_maker/test_video2.mp4")