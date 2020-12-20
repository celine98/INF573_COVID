# INF573 Project

In this project, we implement a Covid social distancing detector.
Given an image, we detect all the people in it, estimate the distance between them, and check whether they are wearing masks (using face and mask detectors).
The model outputs an image annotated with the detected information, and prints out that information in the command prompt. 


For more information please refer to the project report. 

# set-up

## required packages:

	Torch

	opencv-python (cv2)
	
	Pillow

	Cython

	tensorflow
	
## Model download

Before running the code, please download the human detection model and place it in the models/ folder: https://drive.google.com/file/d/1OIjWClONRY96-lC96srLaoPZOq8iM9W2/view?usp=sharing


# Running the code

Run the following command in the project repository:

	main.py [-h] -i IMAGE [-sd SAFE_DISTANCE] [-fl FOCAL_LEN]

**optional arguments:**

	  -h, --help            show this help message and exit
	  
	  -i IMAGE, --image IMAGE
				path to test image
				
	  -sd SAFE_DISTANCE, --safe_distance SAFE_DISTANCE
				Desired safe distance in meters (default = 1m)
				
	  -fl FOCAL_LEN, --focal_len FOCAL_LEN
				Focal Length if known (default = 0.965)

**Example:**

	python3 main.py -i photos/people.png

