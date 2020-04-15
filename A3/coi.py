# Princeton COS 429, Fall 2019

import cv2
import matplotlib.pyplot as plt
import numpy as np

from rect import rect2int, rectDraw, rectEnlarge, rectIntersect, rectSize


def coi2imcoord(coi, c):
    """
    % coi2imcoord   convert coi image coord or rect c to image coord/rect 
    %
    % imc = coi2imcoord(coi, meic)
    %
    % the output coords imc can be use to access actual pixels
    %
    """
    return np.r_[c[:2] + coi.origin[0], c[2:] + coi.origin[1]]

def coiCut(coi, rect, safety_border=0):
    """
    % coiCut   cut a rectangular region from ME image coi 
    %
    % ccoi = coiCut(coi, rect, safety_border)
    %
    % The rect is first rounded (according to our rect rules). 
    % Use safety_border if you belive the edge of the image might be suspect, and 
    % should not be used.  If rect is larger than imageRect-safety_border, the 
    % intersection area is returned.
    %
    """
    rect = rectIntersect(rect, rectEnlarge(coiImageRect(coi), -1 * safety_border))

    if rectSize(rect).prod() > 0:
        imrect = rect2int(coi2imcoord(coi, rect))
        origin = 1 - im2coicoord(coi, imrect[[0, 2]])
        im = coi.im[(imrect[2] - 1):imrect[3], (imrect[0] - 1):imrect[1]]
    else:
        im = np.zeros((0, 0))
        origin = [0, 0]

    return coimage(im, origin, '{}_c'.format(coi.label), coi.level)

def coiGradient(coi):
    """
    % coiGradient   compute dx and dy derivatives of coimage, opt: blur, apply nonlinearity
    %
    %
    % [coidx,coidy] = coiGradient(coi, params...)
    % parameters used:
    %
    % dfilter - apply this filter to compute dx derivative
    %               transpose to compute dy (def. [-1 0 1]/2)
    %               other resonable choices might be:
    %                   [ -1 0 1; -2 0 2; -1 0 1]/8;
    %                 or more generally:
    %                   f = gaussian([-3:3], 0, 2)'*gaussDeriv([-2:2], 0, 1, 1);
    %                   f = f./sum(abs(f(:)));
    %                 with appropriate extent and sigmas
    %            note: this need not be a 1st derivative filter... 
    %
    %
    """
    dfilter = np.array([[-1, 0, 1]]) / 2
    dx = cv2.filter2D(coi.im, ddepth=-1, kernel=dfilter, borderType=cv2.BORDER_CONSTANT)
    dy = cv2.filter2D(coi.im, ddepth=-1, kernel=dfilter.T, borderType=cv2.BORDER_CONSTANT)

    b = int(max(np.floor(np.array(dfilter.shape) / 2)))
    dx = dx[b:-b, b:-b]
    dy = dy[b:-b, b:-b]
    orig_off = b

    coidx = coimage(dx, coi.origin - orig_off, '{}_dx'.format(coi.label), coi.level)
    coidy = coimage(dy, coi.origin - orig_off, '{}_dy'.format(coi.label), coi.level)

    return coidx, coidy

def coiImageRect(coi):
    """
    % coiImageRect   get imageRect from coimage 
    %
    % rect = coiImageRect(coi) 
    %   rect is [left right bottom top]
    """
    s = coi.im.shape
    o = coi.origin
    return np.array([1, s[1], 1, s[0]]) - np.array([o[0], o[0], o[1], o[1]]) + np.array([-0.5, 0.5, -0.5, 0.5])

def coimage(im, origin, label, level=np.inf):
    """
    % coimage   make a  coordiante image from an image array
    %
    % coi = coimage(im, origin, label, level)
    %
    % im - image array
    % origin - location of the image origin in im coordinates
    %     (note: im coordinates are matlab coord, which means the 
    %      lower left corner is <x,y>=[1,1])
    %
    % label - (o) a string to identify the image (e.g. filename, etc.) (def. 'image')
    % level - (o) level number (def. inf == unknown)
    % 
    % coi - a struct that looks something like this:
    %            im: [480x640 double]
    %        origin: [307 271]
    %         label: 'data019_00010_L0'
    %         level: 0
    %
    %
    % Example:  the function coiReduce which changes label & level but not 
    % timestamp or meta fields would call coimage to create the reduced
    % image like this:
    %
    % new_coi = coimage(new_im, new_origin, new_label, in_level+1)
    """
    return CoordinateImage(im, np.array(origin), label, level)

def coiPixCoords(coi):
    """
    % coiPixCoords   compute the x,y coordinates of each pixel 
    %
    % [cx,cy] = coiPixCoords(coi)
    %
    % [cx, cy] - are matricies of the same size as coi.im -- similar to meshgrid
    """
    r = coiImageRect(coi)
    cx, cy = np.meshgrid(np.arange(r[0] + 0.5, r[1] - 0.5 + 1), np.arange(r[2] + 0.5, r[3] - 0.5 + 1))
    return cx, cy

def coiReduce(coi):
    """
    % coiReduce   blur and subsample image by factor of 2
    % 
    % coi = coiReduce(coi, params...)
    % 
    % coi - me image (see coimage)
    % 
    % used params:
    %   'reduce_filt' - filter to use for blurring (e.g. [1 2 1]). The filter
    %                   will be normalized and used in both directions (x&y).  
    """
    reduce_filt = np.array([[1, 2, 1]])
    reduce_filt = reduce_filt.T @ reduce_filt

    baselabel = coi.label[:-3] if coi.label[-3:-1] == '_L' else coi.label
    label = '{}_L{}'.format(baselabel, coi.level + 1)
      
    offset = np.logical_not(np.mod(coi.origin, 2)) + 1

    ssim = reduce_(coi.im, reduce_filt, offset)

    return coimage(ssim, np.ceil(coi.origin / 2), label, coi.level + 1)

def coPyramid(coi, levs=4):
    """
    % coPyramid   make a coordinate pyramid from and coimage with L levels 
    % 
    % mep = mePyramid(coi, levs)
    % 
    % coi - coimage or a cell array of co images (see coimage) 
    %       If cell array of multiple images, then the images should represent
    %       different levels of the same scene.  For example, coi could be a 
    %       previously built coPyramid. 
    %
    % levs - level range to return as [base top] or 
    %           just [top] which is equal to [lowest_input top] (def. [4])
    %
    % cop  - array of images where cop[0] is the bottom most level given 
    %        (original image)
    """
    if not isinstance(coi, list):
        coi = [coi]

    inlevs = []
    for i in range(len(coi)):
        inlevs.append(coi[i].level)
    inlevs = np.array(inlevs)
    np.sort(inlevs)

    levs = [min(inlevs), levs]

    # % these are used to build the initial pyramid. Some levels may be killed 
    # % later if the range of inlevs is greater then the requested levels
    minlev = min(levs[0], min(inlevs))
    maxlev = max(levs[-1], max(inlevs))

    # % make initial pyramid by placing input image into the pyramid and 
    # % reducing them to make higher levels
    cop = []
    for lev in range(minlev, maxlev + 1):
        mlevi = np.argwhere(inlevs == lev)
        if len(mlevi) > 0:
            cop.append(coi[mlevi[0, 0]])
        elif lev > minlev and len(cop) > 0:
            cop.append(coiReduce(cop[-1]))
        else:
            cop.append([])

    # % finally remove levels that are out of the 'levs' range
    cop = cop[(levs[0] - minlev + 1) - 1:levs[1] - minlev + 1]
    return cop

class CoordinateImage:
    def __init__(self, im, origin, label, level):
        self.im = im
        self.origin = origin
        self.label = label
        self.level = level

def drawFaceSequence(fs, from_, step, number, rects=[]):
    """
    % drawFaceSequence  
    %
    % drawFaceSequence(fs, from, step, number, rects)
    """
    plt.ion()

    ####################
    # TODO - part 1c
    for i in range(from_, number):
        # piazza says frames 0, 5,... 45 so those are what we'll use
        # otherwise would have to do i*step - 1 for the coimg to adjust for zero indexing
        coi1 = fs.readImage(i*step)
        rectUno = rects[i]
        imageco(coi1)
        rectDraw(rectUno)
        plt.pause(0.2)
        plt.clf()
    ####################

    plt.ioff()

class FaceSequence:
    """
    %  FaceSequence  class handles reading in data from the sequences
    %    downloaded from http://vision.ucsd.edu/~bbabenko/data/miltrack/
    % 
    % to initialize:
    %  fs = FaceSequence('path/to/data/girl')
    %
    % Now to access the images, you can (method 1) 
    %  coi = fs.readImage(4);
    % will read the 5th image
    % and read it into a coimage struction (help coimage)
    % If you want to access a sequence of images, say every 3rd image starting
    % from the 2nd, do (method 2)
    %   fs.next=1
    %   fs.step=3
    %   coi1 = fs.readNextImage()
    %   coi2 = fs.readNextImage()
    %   so on.. 
    % Additionally, fs contains the "ground truth" rectangles stored with the
    % clip in 
    %    rect = fs.gt_rect(1,:); % rectangle for 1st image
    % Beware: only some frames have valid ground truth rectangles, otherwise
    % this rect = [0 0 0 0]
    """
    def __init__(self, dpath_in):
        self.dpath = dpath_in
        name = dpath_in.name
        self.image_pattern = str(self.dpath / 'imgs' / 'img{:05d}.png')
        irng = np.loadtxt(self.dpath / '{}_frames.txt'.format(name), dtype='int', delimiter=',')
        self.im_number = list(range(irng[0], irng[1] + 1))
        self.step = 1
        self.next = 0
        R = np.loadtxt(self.dpath / '{}_gt.txt'.format(name), delimiter=',')
        self.gt_rect = np.stack((R[:, 0], R[:, 0] + R[:, 2], R[:, 1], R[:, 1] + R[:, 3]), axis=1) - 0.5
        self.gt_good = np.argwhere(R.sum(axis=1) > 0)

    def readImage(self, idx):
        if idx < 0 or idx >= len(self.im_number):
            raise Exception('readImage: index out of range')
        number = self.im_number[idx]
        im = cv2.imread(self.image_pattern.format(number))[:, :, 0] / 255.0
        coi = coimage(im, (0, 0), 'img{:05d}'.format(number), 1)
        return coi

    def readNextImage(self):
        coi = self.readImage(self.next)
        self.next += self.step
        return coi

def im2coicoord(coi, c):
    """
    % im2coicoord   convert image coord or rect c to coi coord/rect 
    %
    % cout = im2coicoord(coi, cin)
    %
    % coi - coimage
    % cin - matrix where each row is either a coord [x y], a rect [l r b t], 
    %       or a single index [ idx ] into the image 
    %       (so [y, x] = ind2sub(im, idx))
    %
    % example: to find the position of the max pixel value of coi, do
    % [v, idx] = max(coi.im(:));
    % c = im2coicoord(coi, idx);
    """
    return np.r_[c[0] - coi.origin[0], c[1] - coi.origin[1]]

def imageco(coi):
    """
    % imageco  displays an coimage in grayscale   
    %
    % 
    % coi - coimage (or mePyramid or matlab image) 
    %
    """
    X = np.arange(1, coi.im.shape[1] + 1) - coi.origin[0]
    Y = np.arange(1, coi.im.shape[0] + 1) - coi.origin[1]
    plt.imshow(coi.im, extent=(X[0], X[-1], Y[-1], Y[0]), cmap='gray')
    plt.gca().set_title(coi.label)

def imagecosc(coi):
    """
    % imagecosc  display an me image in pseudocolor 
    %
    % 
    % coi - coimage (or coPyramid or matlab image) 
    """
    X = np.arange(1, coi.im.shape[1] + 1) - coi.origin[0]
    Y = np.arange(1, coi.im.shape[0] + 1) - coi.origin[1]
    plt.imshow(coi.im, extent=(X[0], X[-1], Y[-1], Y[0]), cmap='viridis')

def reduce_(im, filt, offset=[2, 2]):
    """
    % reduce   blur and subsample by a factor of 2
    %
    %  outim = reduce(im, filt, offset) 
    %
    %  im       input image
    %  filt     (o) filter to use for blurring (def [1 4 6 4 1]) ([] to use def)
    %  offset   (o) offset for subsampling in [x y] i.e. subsamp=(x:2:end) (def [2 2])
    """
    filt = filt / filt.sum()
    tim = cv2.filter2D(im, ddepth=-1, kernel=filt, borderType=cv2.BORDER_CONSTANT)
    outim = tim[(offset[1] - 1)::2, (offset[0] - 1)::2]
    return outim
