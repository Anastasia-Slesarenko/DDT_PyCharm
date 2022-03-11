import numpy as np
import cv2
import multiprocessing as mp
from os import path, remove, makedirs
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import imutils


def rotate(path_file_coord):
    f = open(path_file_coord, mode='r', encoding='utf8', newline='\r\n')
    coord_p = []
    for i, line in enumerate(f):
        coord_p += [list(map(int, line.split(',')))]
    y_start = coord_p[0][1]
    y_0 = coord_p[1][1]
    x_end = coord_p[0][0]
    x_0 = coord_p[1][0]

    dx = x_end-x_0
    dl = np.sqrt((y_start-y_0)**2 + (x_end-x_0)**2)
    sin_theta = dx/dl

    return sin_theta


# Функция для быстрой загрузки кадров искр из видео
def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),  # write out the bar
    sys.stdout.flush()  # flush to stdout


def extract_frames(video_path, frames_dir, file_coord, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """
    sin_theta = rotate(path_file_coord=file_coord)
    theta = np.arcsin(sin_theta) / np.pi * 180
    video_path = path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = path.split(video_path)  # get the video path and filename from the path

    assert path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # lets loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count

            #save_path = frames_dir + "/{:010d}.jpg".format(frame)
            save_path = path.join(frames_dir, video_filename, "{:010d}.jpg".format(frame))  # create the save path
            if not path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                image = imutils.rotate(image, angle=theta)
                cv2.imwrite(save_path, image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_path, frames_dir, file_coord, overwrite=False, every=1, chunk_size=1000):
    """
    Extracts the frames from a video using multiprocessing
    :param file_coord:
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = path.split(video_path)  # get the video path and filename from the path
    # make directory to save frames, its a sub dir in the frames_dir with the video name
    makedirs(path.join(frames_dir, video_filename), exist_ok=True)

    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away

    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV + ffmpeg installation")
        return None  # return None

    frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)  # make sure last chunk has correct end frame, also handles case chunk_size < total

    prefix_str = "Extracting frames from {}".format(video_filename)  # a prefix string to be printed in progress bar

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:

        futures = [executor.submit(extract_frames, video_path, frames_dir, file_coord, overwrite, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)

        for i, f in enumerate(as_completed(futures)):  # as each process completes
            print_progress(i, len(frame_chunks)-1, prefix=prefix_str, suffix='Complete')  # print it's progress

    # return path.join(frames_dir, video_filename)  # when done return the directory containing the frames


if __name__ == '__main__':
    # test it
    discharges = video_to_frames(video_path=r'silicone SPBU/PKD_02.02.22_part/1/cap_cut_crop.mp4',
                                 frames_dir=r'silicone SPBU/PKD_02.02.22_part/1/test_frames',
                                 file_coord=r'silicone SPBU/PKD_02.02.22_part/1/coord_p.txt',
                                 overwrite=True, every=1, chunk_size=1000)
    drops = video_to_frames(video_path=r'silicone SPBU/PKD_02.02.22_part/1/cap_cut_crop_drops.mp4',
                            frames_dir=r'silicone SPBU/PKD_02.02.22_part/1/test_frames_drops',
                            file_coord=r'silicone SPBU/PKD_02.02.22_part/1/coord_d.txt',
                            overwrite=True, every=1, chunk_size=1000)
