# Princeton COS 429, Fall 2019

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def rect2int(rect, offset=(0, 0)):
    """
    % rect2int  round the coordinates of the rect for addressing pixels 
    %
    % irect = rect2int(rect, offset)
    %
    % offset - (o) offset of each pixel from the grid [x y] (def. [0 0])
    %              only the non-integer component of x,y are used, so 
    %              just pass in mei.origin when that is non-integer.  
    %
    % "integer" rects are rects where the rect edge falls on the center of each pixel 
    % (not the edge of each pixel, as the standard rect does). Thus 
    % rectSize(irect) is actually 1 smaller than the actual size of the rect.  
    % irect is typically used to iterate over or cut out the actual pixels 
    % (i.e. irect(1):irect(2)).
    %
    % This function rounds to the actual pixels that make up the rectangle. A pixels has
    % to be at least 50% inside the rect in order to make it (this may not be strictly
    % true for the corners...).
    %
    % This does:  irect = roundto(rect+[0.5 -0.5 0.5 -0.5], offset([1 1 end end]));
    %
    % Note: this means that if offset is non-integer, the resulting irect also 
    % won't have integer values - but mei2imcoord(mei, irect) will allways be integer...
    %
    % Special Case: if the input rect is already an integer rect, it is not touched. 
    %
    """
    def roundto(a, fracto):
        f = np.mod(fracto, 1)
        return np.floor(((a - f) + 0.5) + f).astype(np.int)

    if np.all(np.round(rect) == rect):
        return rect

    return roundto(rect + np.array([0.5, -0.5, 0.5, -0.5]), np.array([offset[0], offset[0], offset[1], offset[1]]))
def rect2uvs(rect1, rect2):
    """
    % rect2uvs  compute uvs from 2 rectangles
    %
    % mot = rect2uvs(r1, r2)
    %
    % r1 -> r2 -  the rectangles 
    %
    % if the 2 dimentions of scale are not the same, then their scale is averaged...
    """
    c1 = rectCenter(rect1)
    c2 = rectCenter(rect2)
    s1 = rectSize(rect1)
    s2 = rectSize(rect2)

    ####################
    # TODO - part 1b
    # round the rects first 
    # plug both sets of coords into:
        # x' = x + (x-x0) * s
        # y' = y + (y-y0) * s
    # to solve for s
    # then use calculated s and coords into:
        # x' = x + u + (x-x0) * s
        # y' = y + v + (y-y0) * s
    # to solve for u and v
    # handle the case where the scale needs to be averaged??

    scaleX = (s2[0] - s1[0]) / s1[0]
    scaleY = (s2[1] - s1[1]) / s1[1]

    if scaleX is not scaleY:
        scale = (scaleX + scaleY) / 2
    else:
        scale = scaleX


    u1 = rect2[0] - rect1[0] - ((rect1[0] - c1[0]) * scale)
    u2 = rect2[1] - rect1[1] - ((rect1[1] - c1[0]) * scale)
    u = (u1+u2) / 2

    v1 = rect2[2] - rect1[2] - ((rect1[2] - c1[1]) * scale)
    v2 = rect2[3] - rect1[3] - ((rect1[3] - c1[1]) * scale)
    v = (v1+v2) / 2

    mot = [u, v, scale, c1[0], c1[1]]
    ####################

    return mot

def rectCenter(rect):
    """
    % rectCenter  returns center of rect(s) r
    %
    % c = rectCenter(r)
    """
    return np.array([rect[:2].mean(), rect[2:].mean()])

def rectChangeLevel(rect, oldlev, newlev):
    """
    % rectChangeLevel   change the pyramid level of the rect 
    %
    % nrect = rectChangeLevel(rect, oldlev, newlev)
    % 
    """
    return (2**(oldlev - newlev)) * rect

def rectDraw(rect):
    """
    % rectDraw   draw rectangle(s) on top of an image
    %
    % h = rectDraw(rect ,col, rectangle_params...)
    %
    % rect - format [left right bottom top] (can be matrix with multiple rows of rects)
    % col - (o) color of the line either [r g b], number, or char (see
    %           toColor).  You can also set the color for each rect by
    %           a 5th column of rect ([l r b t color]). (def. 'g')
    % rectangle_params... - (o) sent in to the matlab function rectangle
    %
    % The rectangles are allways plotted on top of the current figure -- there is 
    % no need to call a "hold on" beforehand.
    %
    % Example:
    %   rectDraw([-10 10 15 22]);
    """
    ax = plt.gca()
    w = max(0, rect[1] - rect[0])
    h = max(0, rect[3] - rect[2])
    ax.add_patch(patches.Rectangle((rect[0], rect[2]), w, h, edgecolor='g', facecolor='none'))

def rectEnlarge(rect, margin):
    """
    % rectEnlarge  change the rectangle rect by adding a margin to it
    %
    % rect = rectEnlarge(rect, margin)
    %
    % margin may be 1x1 (all sides), 1x2 (horiz, vert), or 1x4 in length
    % positive means enlarge,  negative means shrink ([] means do nothing)
    %
    % rect = rectEnlargeFac(rect, fac)
    % after margin has been added to rect.
    %
    % see also rectEnlargeFac (the difference beteween the 2 is the order)
    """
    return rect + np.array([-margin, margin, -margin, margin])

def rectIntersect(rect1, rect2):
    """
    % rectIntersect  intersect 2 rects 
    %
    % rect = rectIntersect(rect1, rect2)
    % where each rect{1,2} may be either 1x4 or nx4. Result is nx4. 
    % or 
    % rect = rectIntersect(rects)
    % where rects = [nx4] matrix of rectangles. Result is 1x4.
    """
    rect = np.zeros_like(rect1)
    rect[[0, 2]] = np.maximum(rect1[[0, 2]], rect2[[0, 2]])
    rect[[1, 3]] = np.minimum(rect1[[1, 3]], rect2[[1, 3]])
    return rect

def rectSize(rect):
    """
    % rectSize   size of the rect
    %
    % s = rectSize(rect)
    % rect can have multiple rows, i.e. (:,[l r b t])
    % s = [w h];
    %
    % remark: returns 1 less than the real size for int rects. For those, use
    % rectSizeInt() instead.
    """
    return np.array([rect[1] - rect[0], rect[3] - rect[2]])
