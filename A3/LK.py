# Princeton COS 429, Fall 2019

import matplotlib.pyplot as plt
import numpy as np

from coi import coiCut, coiGradient, coimage, coiImageRect, coiPixCoords, coPyramid, imageco, imagecosc
from rect import rect2int, rectCenter, rectChangeLevel, rectEnlarge, rectSize 
from uvs import uvs2String, uvsBackwardWarp, uvsChangeLevel, uvsRWarp, uvsWarp


def defineActiveLevels(prect, mep1, mep2, params):
    """
    % defines the levels on which we need to operate...
    % the rule is: there must be at least params.min_pix(1) 
    % pixels to work on 
    """
    s = rectSize(prect)

    ####################
    # TODO - part 3
    newSize = s[0]*s[1]
    maxLevel = 0
    thold = params['min_pix'][0]
    i = 0
    while (newSize >= thold):
        maxLevel = maxLevel + 1
        if (maxLevel >= len(mep1) or maxLevel >= len(mep2)):
            break
        if ((mep1[i].im.shape[0]*mep2[i].im.shape[1]) <= thold):
            break
        i = i+1
        newSize = newSize/4
    minlev = 1
    maxlev = maxLevel
    ####################

    Ls = list(range(minlev, maxlev + 1))
    return Ls

def LKinitParams(varargin={}):
    """
    % LKinitParams  init parameters for scale Lukas-Kanade algo 
    %
    % params = LKinitParams(params...)
    % 
    % params that exist in the input struct are not changed.
    % So to initialize parameters to non-default values,
    % just change those fields either before or after calling this function
    %
    % for a list of parameters and default values, just call this function with 
    % no arguments.
    """
    params = {}

    # % do_scale - if false, don't compute scale parameter
    params['do_scale'] = True

    # %max_iter controls the maximum number of iterations per level
    params['max_iters'] = 40

    # %show_fig is a figure number if show_fig~=0, then the tracker will make a figure(show_fig+level-1) for each level
    params['show_fig'] = 0

    # % the maximum range of levels in which we should track
    params['max_lev_rng'] = (1, 4)

    # % minimum number of pixels needed for doing alignment.
    # % if 2-vector [a b], then b is min pixels to do scale estimation.
    # % if num-pix in level is x, where a<x<b, then only translation is done (do_scale==false).
    params['min_pix'] = (16, 25)
    
    # % lower threshold for error increase from one iter to another
    params['err_change_thr'] = 0

    # % min threshold for a motion change during an iteration to be significant
    # % if motion is less than this in all 3 DOFs, level is considered converged
    params['uvs_min_significant'] = (0.005, 0.005, 0.0005)

    for field, value in varargin.items():
        params[field] = value

    return params

def LKonCoImage(prevcoi, curcoi, prect, init_mot, params):
    """
    % LKonCoImage   find the motion of rect from prev image to cur image
    % 
    % [mot, err] = LKonCoImage(prevcoi, curcoi, prect, init_mot, params)
    %
    % single pyramid level of Lukas-Kanade    
    %
    % mot - final motion
    % err - error of final motion
    """
    max_iters = params['max_iters']
    mot = init_mot
    pmot = mot
    perr = np.inf
    err = np.inf
    ps = prevcoi.im.shape

    # % cut out relevant rectangle (last argument is border for dx,dy)
    pcut1 = coiCut(prevcoi, prect, 1)

    # % do tracking iterations
    exitcond = 'many_iter'
    spcutlast = 0
    for iter_ in range(max_iters):

        # %warp 'curent' (2nd) image according to motion
        wcoi = uvsBackwardWarp(mot, curcoi, coiImageRect(pcut1))

        # % make sure that the 2 images are the same by cutting pcut more
        # % in case wcoi is smaller (because it is slightly out of the image)
        pcut = coiCut(pcut1, coiImageRect(wcoi))

        # %compute A'A when needed:
        # %we compute A'A on first iteration or when cut rectangle has changed 
        # % (due to warp near the border of the image)
        if iter_ == 0 or np.any(np.array(pcut.im.shape) != spcutlast):
            spcutlast = np.array(pcut.im.shape)
            # % check if we still have enough pix to track...
            if spcutlast[:2].prod() < params['min_pix'][-1] or np.any(spcutlast[:2] < 3):
                exitcond = 'out_of_image'
                break

            #  %set up variables for easy access
            # %pcrect = coiImageRect(pcut); % previous cut rect 
            # %pcrs = rectSize(pcrect); % size of the previous cut rect
            npix = pcut.im.size

            # %  compute derivative images... (these images will have same size as pcut)
            prevcoich = coiCut(prevcoi, rectEnlarge(coiImageRect(pcut),1))
            dx, dy = coiGradient(prevcoich)

            if not params['do_scale']:
                ####################
                # TODO - part 2a
                aFlat = dx.im.flatten()
                bFlat = dy.im.flatten()
                A = np.zeros((aFlat.shape[0], 2))
                A[:,0] = aFlat
                A[:,1] = bFlat
                ####################
            else:
                ####################
                # TODO - part 2b
                aFlat = dx.im.flatten()
                bFlat = dy.im.flatten()
                A = np.zeros((aFlat.shape[0], 3))
                A[:,0] = aFlat
                A[:,1] = bFlat
                cx, cy = coiPixCoords(pcut)
                xdiff = cx - mot[3]
                xdiff = xdiff.flatten()
                ydiff = cy - mot[4]
                ydiff = ydiff.flatten()

                xdot = aFlat * xdiff
                ydot = bFlat * ydiff
                total = xdot + ydot
                A[:,2] = total
                ####################


            ####################
            # TODO - part 2a
            AtA = A.transpose() @ A
            AtAinv = np.linalg.inv(AtA)
            ####################

        # %compute Atb 
        it = pcut.im - wcoi.im

        ####################
        # TODO - part 2a
        b = np.array([it.flatten()]).transpose()
        Atb = A.transpose() @ b
        xx = AtAinv @ Atb
        # flatten to get in simpler, 1d form
        mot_update = xx.flatten()
        ####################

        err = np.power(it.flatten(), 2).sum() / npix

        if params['show_fig']:
            plt.ion()
            fig = plt.figure(params['show_fig'])
            plt.clf()
            ax1 = plt.subplot(3, 1, 1)
            imageco(pcut)
            ax1.set_title('Previous L={}'.format(prevcoi.level))
            ax1.plot(mot[3], mot[4], 'r+')

            ax2 = plt.subplot(3, 1, 2)
            imageco(wcoi)
            ax2.set_title('Warp Current I={}\nuvs {}'.format(iter_, uvs2String(mot, False)))
            ax2.plot(mot[3], mot[4], 'r+')

            ax3 = plt.subplot(3, 1, 3)
            imagecosc(coimage(it, pcut.origin, 'ssd'))
            ax3.set_title('err={:.6f}'.format(err))

            plt.tight_layout()
            print('  Err {}: {:.6f}'.format(iter_, err))
            plt.pause(0.2)

        # %exit condition: error increased
        if err - perr > params['err_change_thr']:
            mot = pmot
            err = perr
            exitcond = 'inc_error'
            break

        perr = err
        pmot = mot

        # %update motion
        motNew = mot.copy()

        if not params['do_scale']:
            ####################
            # TODO - part 2a
            motNew[0] = mot[0] + mot_update[0]
            motNew[1] = mot[1] + mot_update[1]
            ####################
        else:
            ####################
            # TODO - part 2b
            motNew[0] = mot[0] + mot_update[0] + (mot_update[0] * mot[2])
            motNew[1] = mot[1] + mot_update[1] + (mot_update[1] * mot[2])
            motNew[2] = mot[2] + mot_update[2] + (mot_update[2] * mot[2])

            ####################

        mot = motNew

        if params['show_fig']:
            print('  New mot {}: {}'.format(iter_, uvs2String(mot)))

        if (abs(pmot[0] - mot[0]) < params['uvs_min_significant'][0] and
                abs(pmot[1] - mot[1]) < params['uvs_min_significant'][1] and
                abs(pmot[2] - mot[2]) < params['uvs_min_significant'][2]):
            exitcond = 'insig_mot'
            break

    if params['show_fig']:
        plt.gca().set_title('err={:.6f}\n{}'.format(err, exitcond))
        print('  Final Lev Mot: {} iters: {} exit: {}'.format(uvs2String(mot), iter_ + 1, exitcond))
        plt.ioff()

    return mot, err

def LKonPyramid(prevPyr, curPyr, prect, init_mot, varargin):
    """
    % LKonPyramid perform Lukas-Kande on multiresolution image pyramid
    %
    % mot = LKonPyramid(prevPyr, curPyr, prect, init_mot, params)
    %
    % finds the image within rectangle prect of the previous frame (prevPyr) in 
    %   the current frame (curPyr), using Lukas-Kanade tracking within the pyramid
    % mot is a uvs vector [x y scale x_0 y_0] (see functions uvs* )
    % params is defined by LKinitParams
    %
    % Note: All motions (input and output) are expressed in the base level of
    % the input pyramids (prevPyr{1}.level).
    %
    % features:
    %    if params.do_scale==t, then scale is optimized, else it is kept equal
    %       to init_mot's scale
    %
    %    debugging - if params.show_fig~=0, then we open figures for each level and
    %                show the motion and residual error after each iteration 
    %
    % this calls LKonCoImage for each level
    """
    params = LKinitParams(varargin)

    assert prevPyr[0].level == curPyr[0].level, 'Pyramid base levels must be the same.'

    # %what levels of the pyramid should we run over
    Ls = defineActiveLevels(prect, prevPyr, curPyr, params)

    mot = init_mot.copy()
    plev = prevPyr[0].level

    for lev in Ls[::-1]:
        mot = uvsChangeLevel(mot, plev, lev)
        lprect = rectChangeLevel(prect, 1, lev)
        cparams = params.copy()

        # % if num pixels is less then min_pix(2), estimate only u,v on this lev
        if np.prod(rectSize(lprect)) < params['min_pix'][-1]:
            cparams['do_scale'] = False

        if params['show_fig']:
            cparams['show_fig'] = lev
            print('Lev {} in {} : {}'.format(prevPyr[lev - 1].level, uvs2String(mot), rect2int(lprect)))
        plev = lev

        # % track in this level
        mot, err = LKonCoImage(prevPyr[lev - 1], curPyr[lev - 1], lprect, mot, cparams)
    plt.show()

    # % change motion back to level 1 (in case last level was not 1)
    # % Note: this 'lev' is the index of level (as in pyr{lev}, not
    # % neccessarily the real level (as in pyr{lev}.level).
    # % Thus the motion is always given in the base level of the pyrmaid.
    mot = uvsChangeLevel(mot, plev, 1)
    return mot

def LKonSequence(fs, seq_length=20, varargin={}):
    """
    % LKonSequence  run Lukas-Kanade using incremental tracking on a sequence
    %
    % rects = LKonSequence(fs, varargin)
    % fs - is a FaceSequence class. Set fs.next and fs.step to define the
    %      frames 
    """

    # % initial rect in first frame
    init_rect = fs.gt_rect[fs.next]

    # % initial motion from frame 0->1 
    init_mot = np.r_[[0, 0, 0], rectCenter(init_rect)]

    lk_params = LKinitParams(varargin)

    # % init rects - first line is frame 0...length
    rects = np.tile(init_rect, (seq_length + 1, 1))

    # % init mots - rows are 0->1, 1->2, ..., (length-1 -> length)
    mots = np.tile(init_mot, (seq_length + 1, 1))

    prevPyr = coPyramid(fs.readNextImage())

    for i in range(seq_length):
        curPyr = coPyramid(fs.readNextImage())

        # % now call LK and fill in result rects and mots, make prediction for
        # % next motion (making some assumption that the motion is smooth) 
        # % note: recall that the x0,y0 componenets of the init_mot 
        # % should be near the prect center 

        ####################
        # TODO - part 4 

        mot = LKonPyramid(prevPyr, curPyr, init_rect, init_mot, varargin)
        mots[i+1,:] = mot
        init_rect = uvsRWarp(mot, init_rect)
        rects[i+1, :] = init_rect
        init_mot = np.array([mot[0]*0.7, mot[1]*0.7, mot[2]*0.7, rectCenter(init_rect)[0], rectCenter(init_rect)[1]])
        ####################

        print('Tracking {} to image {} motion {} rect {}'.format(i + 1, curPyr[0].label, uvs2String(mot), rects[i + 1, :]))
        prevPyr = curPyr
        #init_rect = rects[i]

    return rects
