# Princeton COS 429, Fall 2019

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from coi import coPyramid, drawFaceSequence, FaceSequence, imageco
from LK import defineActiveLevels, LKinitParams, LKonCoImage, LKonPyramid, LKonSequence
from rect import rect2uvs, rectCenter, rectDraw
from uvs import uvsChangeLevel, uvsInv, uvsRWarp

np.set_printoptions(suppress=True)


def get_face_sequence(data_dir='data', seq_name='girl'):
    seq_dir = Path(data_dir) / seq_name
    return FaceSequence(seq_dir)

def test(part):
    if part == '1a':
        # % let's define a rect: [xmin xmax ymin ymax]
        rect1 = np.array([145.5228, 241.2306, 63.1484, 158.1256])

        # % and a motion (with center of motion at center of rect1) : mot = [u v s x0 y0] 
        mot = np.r_[[13, 10.5, 0.09], rectCenter(rect1)]

        # % to warp rect1 according to mot, do
        rect2 = uvsRWarp(mot, rect1)
        moti = uvsInv(mot)
        print(moti)
        rect1_new = uvsRWarp(moti, rect2)
        print(rect1_new)
        assert np.abs(rect1_new - rect1).sum() < 1e-10

    elif part == '1b':
        rect1 = np.array([145.5228, 241.2306, 63.1484, 158.1256])
        mot = np.r_[[13, 10.5, 0.09], rectCenter(rect1)]
        rect2 = uvsRWarp(mot, rect1)
        mot2 = rect2uvs(rect1, rect2)
        print(mot2)
        assert np.abs(mot-mot2).sum() < 1e-10

    elif part == '1c':
        fs = get_face_sequence(seq_name='girl')

        #coi1 = fs.readImage(5)
        #fs.next = 1
        #fs.step = 3
        #coi1 = fs.readNextImage()
        #coi2 = fs.readNextImage()

        # % note: for 'girl' the ground truth is defined only for frames 1,6,11,16...
        # % the following should display a movie of frames 1:5:51 with a rect drawn
        # % around the face:
        drawFaceSequence(fs, 0, 5, 10, fs.gt_rect[0:51:5, :])
        plt.show()

    elif part == '2a':
        # % now implement the translation-only part of LKonCoImage.m

        # % test it on frames 1 and 6 of the girl sequence
        fs = get_face_sequence(seq_name='girl')
        fs.next = 0
        fs.step = 5
        prect = fs.gt_rect[fs.next, :]
        init_mot = np.r_[[0, 0, 0], rectCenter(prect)]
        prevcoi = fs.readNextImage()
        curcoi = fs.readNextImage()
        params = LKinitParams()
        params['do_scale'] = False
        params['show_fig'] = True

        # % you should see an animation with the error image becoming relatively blue
        # % (indicating low error)
        mot, err = LKonCoImage(prevcoi, curcoi, prect, init_mot, params)
        plt.show()

    elif part == '2b':
        # % now implement the translation-scale part of LKonCoImage.m

        fs = get_face_sequence(seq_name='girl')

        params = LKinitParams()
        params['show_fig'] = True

        prect = fs.gt_rect[40, :]
        init_mot = np.r_[[0, 0, 0], rectCenter(prect)]
        prevcoi = fs.readImage(40)
        curcoi = fs.readImage(42)

        # % notice in the visualization that single-scale LK does the best it can, but
        # % there is still error around the outside of the face
        params['do_scale'] = False
        mot, err = LKonCoImage(prevcoi, curcoi, prect, init_mot, params)
        plt.show()

        # % now run the (u,v,s) version - notice that the final error is lower, and
        # % that eventually it converges to the right scale
        params['do_scale'] = True
        mot, err, = LKonCoImage(prevcoi, curcoi, prect, init_mot, params)
        plt.show()

    elif part == '3':
        fs = get_face_sequence(seq_name='girl')

        prect = fs.gt_rect[40, :]
        prevcoi = fs.readImage(40)
        curcoi = fs.readImage(42)

        # % create pyramids for prevcoi and curcoi
        prevpyr = coPyramid(prevcoi, 5)
        curpyr = coPyramid(curcoi, 5)

        #imageco(prevpyr[0])
        #imageco(prevpyr[1])
        #imageco(prevpyr[2])
        #imageco(prevpyr[3])
        #imageco(prevpyr[4])

        # % test it - should only return the first 4 levels
        params = LKinitParams()
        params['min_pix'] = (64, 25)
        print(defineActiveLevels(prect, prevpyr, curpyr, params))

        init_mot = np.r_[[0, 0, 0], rectCenter(prect)]
        mot, err = LKonCoImage(prevcoi, curcoi, prect, init_mot, params)

        # % we're going *up* the pyramid, so motion should be smaller
        mot2 = uvsChangeLevel(mot, 1, 2)
        print(mot2)
        mot21 = uvsChangeLevel(mot2, 2, 1)
        print(mot21)
        assert np.abs(mot21 - mot).sum() < 1e-10

        prevpyr = coPyramid(prevcoi, 5)
        curpyr = coPyramid(fs.readImage(50))
        mot = LKonPyramid(prevpyr, curpyr, prect, init_mot, {'show_fig': True})
        print(mot)

    elif part == '4':
        fs = get_face_sequence(seq_name='girl')

        fs.next = 0
        fs.step = 1
        seq_length = 80
        rects = LKonSequence(fs, seq_length)

        plt.ion()
        for i in range(seq_length):
            plt.clf()
            imageco(fs.readImage(i))
            rectDraw(rects[i, :])
            plt.pause(0.2)
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    #parts_to_test = ['1a', '1b', '1c', '2a', '2b', '3', '4']
    parts_to_test = ['1a']
    for part in parts_to_test:
        print('Testing part {}'.format(part))
        test(part)
        print()
