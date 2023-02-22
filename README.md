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



temp - the analyze step 


analyze_shell(video, fps, label='bad', save_path=None, show_plot=True)
This function analyzes a video by computing the mean intensity of each frame and generating a plot of the results. The plot can be saved to a file if desired.
Parameters:
•	video (numpy.ndarray): the input video
•	fps (int): the frame rate of the video
•	label (str): a label for the video (default: 'bad')
•	save_path (str): the path to save the plot image (default: None)
•	show_plot (bool): whether to display the plot (default: True)
Returns:
•	None
crop_vid(video, left, top, right, bottom)
This function crops a video to the specified rectangle.
Parameters:
•	video (numpy.ndarray): the input video
•	left (int): the x-coordinate of the left edge of the rectangle
•	top (int): the y-coordinate of the top edge of the rectangle
•	right (int): the x-coordinate of the right edge of the rectangle
•	bottom (int): the y-coordinate of the bottom edge of the rectangle
Returns:
•	numpy.ndarray: the cropped video
time_crop(video, dur, start=0, fs=24)
This function crops a video to a specified duration and starting frame.
Parameters:
•	video (numpy.ndarray): the input video
•	dur (int): the duration of the cropped video, in seconds
•	start (int): the starting frame of the cropped video ()
time_crop(video, dur, start, fs)
This function crops a segment of a video based on a given duration and start time.
Parameters:
•	video (numpy.ndarray): The input video as a numpy array.
•	dur (int): The duration of the segment to be cropped in seconds.
•	start (int): The start time of the segment to be cropped in seconds.
•	fs (int): The frame rate of the video in frames per second.
Returns:
•	numpy.ndarray: The cropped video segment.
 
analyze_shell2(video, label, fs)
This function analyzes a video segment by calculating its optical flow and extracting features based on it.
Parameters:
•	video (numpy.ndarray): The input video segment as a numpy array.
•	label (str): A label for the video segment (either 'good' or 'bad').
•	fs (int): The frame rate of the video in frames per second.
Returns:
•	numpy.ndarray: The extracted features as a numpy array.
 
crop_vid(video, left, top, right, bottom)
This function crops a rectangular region from a video.
Parameters:
•	video (numpy.ndarray): The input video as a numpy array.
•	left (int): The x-coordinate of the left edge of the rectangular region to be cropped.
•	top (int): The y-coordinate of the top edge of the rectangular region to be cropped.
•	right (int): The x-coordinate of the right edge of the rectangular region to be cropped.
•	bottom (int): The y-coordinate of the bottom edge of the rectangular region to be cropped.
Returns:
•	numpy.ndarray: The cropped video.
 
time_corp_analyze_shell(video, stride, dur, left, top, right, bottom, label='bad', fs=24)
This function analyzes a video by cropping it into multiple segments and running analyze_shell2 on each segment.
Parameters:
•	video (numpy.ndarray): The input video as a numpy array.
•	stride (int): The stride used to separate the segments.
•	dur (int): The duration of each segment.
•	left (int): The x-coordinate of the left edge of the rectangular region to be cropped.
•	top (int): The y-coordinate of the top edge of the rectangular region to be cropped.
•	right (int): The x-coordinate of the right edge of the rectangular region to be cropped.
•	bottom (int): The y-coordinate of the bottom edge of the rectangular region to be cropped.
•	label (str): A label for the video (default: 'bad').
•	fs (int): The frame rate of the video in frames per second (default: 24).
Returns:
•	numpy.ndarray: A list of results of analyze_shell2.
 
draw_grid(img, step_size=20, color=(255, 0, 0), thickness=1)
This function draws a grid on an image.
Parameters:
•	img (numpy.ndarray): The input image as a numpy array.
•	step_size (int): The size of one step (default: 20).
•	color (tuple (B, G, R)): The color of the grid lines (default: (255, 0, 0)).
•	thickness (int): The thickness of the grid lines (default: 1).
Returns:
•	numpy.ndarray: The output image with the grid




