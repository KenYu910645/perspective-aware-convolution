import cv2
import os
from collections import defaultdict

SQU_DIR = "../viz_result/kitti_test_sequence"

image_names = sorted(os.listdir(SQU_DIR))

print(f"Number of image = {len(image_names)}")

squ_dict = defaultdict(list)
for image_name in image_names:
    squ_dict[ image_name.split('_')[0] ].append(image_name)

# Filter unwanted sequence 
new_image_names = []
for image_name in image_names:
    if int(image_name.split("_")[0]) in [1, 4, 5, 18, 20]:
        new_image_names.append(image_name)
image_names = new_image_names


# Get the dimensions of the first image (assuming all images have the same dimensions)
first_image = cv2.imread(os.path.join(SQU_DIR, image_names[0]))
height, width, layers = first_image.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use different codecs, e.g., 'MJPG'
output_video = cv2.VideoWriter(f'../viz_result/video/all_sequence_best.avi', fourcc, 20, (width, height))

for image_name in image_names:
    img = cv2.imread(os.path.join(SQU_DIR, image_name))
    output_video.write(img)

# Release the VideoWriter and close the video file
output_video.release()

print(f"Save video to ../viz_result/video/all_sequence.avi")


# Output every sequence to seperate video
# for squ_name in squ_dict:

#     # Get the dimensions of the first image (assuming all images have the same dimensions)
#     first_image = cv2.imread(os.path.join(SQU_DIR, squ_dict[squ_name][0]))
#     height, width, layers = first_image.shape

#     # Define the codec and create a VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use different codecs, e.g., 'MJPG'
#     output_video = cv2.VideoWriter(f'../viz_result/video/{squ_name}.avi', fourcc, 20, (width, height))
    
#     for image_name in squ_dict[squ_name]:
#         img = cv2.imread(os.path.join(SQU_DIR, image_name))
#         output_video.write(img)

#     # Release the VideoWriter and close the video file
#     output_video.release()

#     print(f"Save video to ../viz_result/video/{squ_name}.avi")