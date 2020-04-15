"""
 Princeton University, COS 429, Fall 2019
"""


def find_faces(img, stride, thresh, params, orientations, wrap180):
    """Find faces in an image

    Args:
        img: an image
        stride: how far to move between locations at which the detector is run,
            at the finest (36x36) scale.  This is effectively scaled up for
            larger windows.
        thresh: probability threshold for calling a detection a face
        params: trained face classifier parameters
        orientations: the number of HoG gradient orientations to use
        wrap180: if true, the HoG orientations cover 180 degrees, else 360

    Returns:
        outimg: copy of img with face locations marked
    """
    hog_input_size = 36
    windowsize = 36
    if stride > windowsize:
        stride = windowsize

    windowsize_org = windowsize
    stride_org = stride

    height, width = img.shape
    probmap = np.zeros([height, width])
    outimg = np.array(img)

    # Loop over windowsize x windowsize windows, advancing by stride
    hog_descriptor_size = 100 * orientations
    window_descriptor = np.zeros([1,hog_descriptor_size + 1])

    # Extract windows of ever-larger sizes, resizing each window before 
    # passing it in to the HoG computation
    while (windowsize <= min(height,width)):

        for i in range(0, width-windowsize, stride):
            for j in range(0, height-windowsize, stride):

                print(height, width)
                print(windowsize, j,i)
                # Crop out a windowsize x windowsize window starting at (i,j)
                crop = img[j:j+windowsize,i:i+windowsize] 
                # resize before passing it into HOG --> make sure 36x36 
                crop = cv2.resize(crop, (hog_input_size, hog_input_size))
                #print(crop.shape)
                print(windowsize,j,i)

                # Compute a HoG descriptor, and run the classifier
                window_descriptor[0,0] = 0
                window_descriptor[0, 1:] = hog36(crop, orientations, wrap180)
                # NEED TO TRAIN AND RUN CLASSIFIER ?? PROB --> FIT () ?? or since trained params good 
                probability = logistic_prob(window_descriptor, params) #or need to do both fit + prob 

                # Mark detection probability in probmap
                win_i = i + int((windowsize - stride) / 2)
                win_j = j + int((windowsize - stride) / 2)
                probmap[win_i:win_i+stride, win_j:win_j+stride] = probability
                
                print(windowsize,j,i)
                # If probability of a face is below thresh, continue 
                # else mark the face on img 
                if probability < thresh:
                    continue

                #print(windowsize)
                # Mark the face in outimg
                outimg[j, i:i+windowsize] = 255
                outimg[j+windowsize-1, i:i+windowsize] = 255
                outimg[j:j+windowsize, i] = 255
                outimg[j:j+windowsize, i+windowsize-1] = 255
                
                print("HH", j,i)

        # scale by 20% each iteration
        windowsize = round(windowsize*1.2)
        stride = round((windowsize)*(stride_org/windowsize_org))
        print("here")


    return outimg
