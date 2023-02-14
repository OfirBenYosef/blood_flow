# blood_flow
Image processing functions

This repository contains a set of Python functions for image processing. The functions can be used for various tasks, such as contrast enhancement, background removal, thresholding, and segmentation of colon tissue.

The following functions are available:

contrast_enhancement(img)

This function takes an input image and applies contrast enhancement using Contrast Limited Adaptive Histogram Equalization (CLAHE) on the green channel of the image.

create_mask(image, points)

This function takes an input image and a set of points, and creates a binary mask of the region defined by the points.

colon_seg(img)

This function takes an input image and applies active contour segmentation to segment the colon tissue in the image.

threshold(image)

This function takes an input image and applies thresholding using Otsu's method to obtain a binary image. It then performs morphological opening to remove small objects and smooth the edges.

F_frangi(img_adapteq)

This function applies Frangi vesselness filter to the input image to enhance the visibility of vessels in the image.

apply_mask(video, mask)

This function takes an input video and a binary mask, and applies the mask to each frame of the video to obtain a masked video.

All of the functions are implemented using the Python OpenCV and scikit-image libraries.

Usage

To use these functions in your project, simply import the relevant function into your Python script and call it with the appropriate input parameters.

