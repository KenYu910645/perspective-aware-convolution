import cv2
import os
from fire import Fire
from collections import defaultdict
from PIL import Image
import os

SQU_SHOW = [1, 5, 18] # [1, 4, 5, 18, 20]

def main(squence_dir, output_path):

    image_names = sorted(os.listdir(squence_dir))

    print(f"Number of image = {len(image_names)}")

    squ_dict = defaultdict(list)
    for image_name in image_names:
        squ_dict[ image_name.split('_')[0] ].append(image_name)

    # Filter unwanted sequence 
    new_image_names = []
    for image_name in image_names:
        if int(image_name.split("_")[0]) in SQU_SHOW:
            new_image_names.append(image_name)
    image_names = new_image_names

    # Get the dimensions of the first image (assuming all images have the same dimensions)
    first_image = cv2.imread(os.path.join(squence_dir, image_names[0]))
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    if output_path.endswith('.gif'):
        # Load each image and append it to the frames list
        frames = []
        for image_name in image_names:
            img = Image.open(os.path.join(squence_dir, image_name))
            frames.append(img)
        
        # Save to gif file
        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=50)
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use different codecs, e.g., 'MJPG'
        output_video = cv2.VideoWriter(output_path, fourcc, 20, (width, height))

        for image_name in image_names:
            img = cv2.imread(os.path.join(squence_dir, image_name))
            output_video.write(img)

        # Release the VideoWriter and close the video file
        output_video.release()

    print(f"Save video to {output_path}")

if __name__ == "__main__":
    Fire(main)