# Princeton COS 429, Fall 2019

import numpy as np

from coi import coi2imcoord, coiImageRect, coimage
from rect import rect2int, rectEnlarge, rectIntersect, rectSize


def uvs2String(mot, show_center=True):
    """
    % uvs2String  format tsuv motion to a string for printing
    %
    % str = uvs2String(mot, show_center)
    % show_center - (o) print center location (def. 1)
    """
    if not show_center:
        return '<{:0.2f} {:0.2f} {:0.3f}>'.format(*mot[:3])
    return '<{:0.2f} {:0.2f} {:0.3f} ({:0.0f},{:0.0f})>'.format(*mot)

def uvsBackwardWarp(mot, mei, destrect):
    """
    % uvsBackwardWarp  warp the image using mot's inverse and cut the result to destrect 
    %
    % wcoi = uvsBackwardWarp(mot, mei, destrect)
    % 
    % if mot is the motion from A -> B, then this function warps a piece of B (input coi)
    % (aka. source) to be aligned with A (aka. dest). The pixels that should exist
    % in the result is defined by rounding destrect, which is in A's coordinate system. 
    %
    % This function just calls:
    %   wcoi = uvsWarp(uvsInv(mot), mei, destrect);
    """
    return uvsWarp(uvsInv(mot), mei, destrect)

def uvsChangeLevel(mot, oldlev, newlev):
    """
    % uvsChangeLevel  change the pyramid level of the motion 
    %
    % nmot = uvsChangeLevel(mot, oldlev, newlev)
    % 
    """

    ####################
    # TODO - part 3
    diff = abs(oldlev - newlev)
    if oldlev > newlev:
        factor = diff * 2
    elif oldlev < newlev:
        factor = 1 / (diff * 2)
    else:
        factor = 1

    nmot = mot * factor
    ####################
    return nmot

def uvsInv(mot):
    """
    % uvsInv  invert the transScaleUV motion 
    %
    % motinv = uvsInv(mot)
    %
    % motinv is defined such that: 
    %    uvsPlus(mot,motinv) -> zero motion with center at mot's center;
    """
    ####################
    # TODO - part 1a
    # x' to x and y' to y is just reverse direction
    u1 = -mot[0]
    v1 = -mot[1]
    
    # w' to w is solved via getting (w-w')/w' in terms of the original s
    sNew = -mot[2]/(mot[2]+1)
    
    # new motion has moved center aka projection of x0, y0 by motion
    x0 = mot[3] + mot[0]
    y0 = mot[4] + mot[1]
    motinv = np.array([u1, v1, sNew, x0, y0])
    ####################

    return motinv

def uvsRBackwardWarp(mot, rect):
    """
    % uvsRBackwardWarp  warp a rect according to inverse of mot
    %
    % wc = uvsRBackwardWarp(mot, r)
    % i.e. uvsRBackwardWarp(mot, uvsRWarp(mot, r))==r
    """
    u, v, s, x0, y0 = mot
    return np.r_[rect[:2] - u + s * x0, rect[2:] - v + s * y0] / (1 + s)

def uvsRWarp(motion, rect):
    """
    % uvsRWarp   warp a rect according to mot
    %
    % wr = uvsRWarp(mot, r)
    %
    % motion - 5 array of motions
    % rect - k array of rects (k is atleast 4).
    %
    """
    left, right, bottom, top = rect
    u, v, s, x0, y0 = motion

    wl = left + u + s * (left - x0)
    wr = right + u + s * (right - x0)
    wb = bottom + v + s * (bottom - y0)
    wt = top + v + s * (top - y0)

    return np.array([wl, wr, wb, wt])

def uvsWarp(mot, coi, destrect, crop_it=True):
    """
    % uvsWarp  warp the image using mot and cut the result to destrect 
    %
    % wcoi = uvsWarp(mot, coi, destrect)
    % 
    % if mot is the motion from A -> B, then this function warps a piece of A
    % (input coi) (aka. source) to be aligned with B (aka. dest). The pixels that
    % should exist in the result is defined by rounding destrect, which is in B's
    % coordinate system.
    %
    % destrect - (o) me rect, and is rounded by rect2int to find the pixels that
    % will exist in the final image.  Note: this means that an "integer" sized and
    % pixel aligned rectangle typically looks like [-1.5 1.5 -2.5 2.5], not [-1 1
    % -2 2]. If not given, the whole input image is warped and the resulting
    % image will be assumed to be integer aligned.
    %
    % crop_it - (o) default: true; if true and if destrect is too large for 
    % warped image, the 2 are intersected - similar to coiCut (Note: it's pixel offset is still used for the final image
    % pixel offset. Also if true, any nan rows/columns are removed (due to
    % input image)
    """
    def resampleMei(coi, srect, drect):
        isrect = coi2imcoord(coi, srect)
        crect = np.array([np.floor(isrect[0]), np.ceil(isrect[1]), np.floor(isrect[2]), np.ceil(isrect[3])]).astype(np.int)
        cutim = coi.im[crect[2] - 1:crect[3], crect[0] - 1:crect[1]]
        cisrect = isrect - np.array([crect[0], crect[0], crect[2], crect[2]]) + 1
        s = rectSize(drect) + 1
        rim = resample1D(cutim, cisrect[2:],  s[1], 1)
        rim = resample1D(rim, cisrect[:2],  s[0], 2)

        return rim

    def resample1D(Ain, source, dests, dim):
        A = Ain.astype(np.float)
        rr = np.linspace(source[0], source[1], dests)
        lp = np.floor(rr).astype(np.int) - 1
        rp = np.ceil(rr).astype(np.int) - 1
        rw = np.mod(rr, 1)
        lw = 1 - rw

        if dim == 1:
            B = A[lp, :] * lw.T[:, np.newaxis] + A[rp, :] * rw.T[:, np.newaxis]
        elif dim == 2:
            B = A[:, lp] * lw + A[:, rp] * rw
        return B

    crect = coiImageRect(coi)
    idestrect = rect2int(destrect)

    if crop_it:
        wirect = uvsRWarp(mot, rectEnlarge(crect, -0.99))
        assert np.all(np.mod(rectSize(destrect), 1) == 0)
        doffset = np.mod(destrect[0:3:2] + 0.5, 1)
        destrect = rect2int(rectIntersect(wirect, destrect))
        idestrect = rect2int(destrect, doffset)

    if np.any(rectSize(destrect) < 1):
        wim = np.zeros((0, 0))
    else:
        srect = uvsRBackwardWarp(mot, idestrect)
        wim = resampleMei(coi, srect, idestrect)
    wcoi = coimage(wim, tuple(1 - idestrect[0:3:2]), '{}_warp'.format(coi.label), coi.level)

    return wcoi
