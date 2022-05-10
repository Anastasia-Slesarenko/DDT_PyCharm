import cv2
from itertools import count
import numpy as np
from moviepy.editor import VideoFileClip

merg = count(1)
merg_light = count(1)
multi_roll = count(1)
multi_roll_light = count(1)

def count_merg(with_light):
    if with_light == 'No':
        next(merg)
    else:
        next(merg)
        next(merg_light)

def count_multi_rolloff(with_light):
    if with_light == 'No':
        next(multi_roll)
    else:
        next(multi_roll)
        next(multi_roll_light)

def get_count_merg():
    return print("Number of mergers: '{}'".format(next(merg)-1)),\
           print("Number of mergers with discharge: '{}'".format(next(merg_light)-1))

def get_count_multi():
    return print("Number of multi-rolloff: '{}'".format(next(multi_roll)-1)),\
           print("Number of multi-rolloff with discharge: '{}'".format(next(multi_roll_light)-1))

if __name__ == '__main__':

    frame_m = []
    frame_m_l = []
    frame_m_r_l = []
    frame_m_r = []

    cap = cv2.VideoCapture(
        r'C:/Users/1/Desktop/VKR/Master/data/silicone SPBU/PKD_02.02.22_part/2/test_frames_drops/video_maker/vid.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 65000)
    while int(cap.get(cv2.CAP_PROP_POS_FRAMES)) < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('fgmask', frame)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            count_merg('No')
            frame_m += [int(cap.get(cv2.CAP_PROP_POS_FRAMES))]
        if k == ord('w'):
            count_merg(1)
            frame_m += [int(cap.get(cv2.CAP_PROP_POS_FRAMES))]
            frame_m_l += [int(cap.get(cv2.CAP_PROP_POS_FRAMES))]

        if k == ord('u'):
            count_multi_rolloff('No')
            frame_m_r += [int(cap.get(cv2.CAP_PROP_POS_FRAMES))]
        if k == ord('i'):
            count_multi_rolloff(1)
            frame_m_r += [int(cap.get(cv2.CAP_PROP_POS_FRAMES))]
            frame_m_r_l += [int(cap.get(cv2.CAP_PROP_POS_FRAMES))]

        if k == ord('p'):
            end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            break
        if k == ord('z'):
            cv2.waitKey()

    #end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cv2.destroyAllWindows()

    print(frame_m, frame_m_l)
    get_count_merg()
    print(frame_m_r, frame_m_r_l)
    get_count_multi()
    print(end_frame)

