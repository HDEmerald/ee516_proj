"""
convert.py
~~~~~~~~~~

A simple script that turns a txt file of the name '*.labeled_faces.txt'
into many individual txt files in a folder called labels. The miriad 
of txt files are each label files for a corresponding image. The label
contains a path to the image (from the given dataset path), the center 
of a bounding box (bbox) around a persons face, as well as the width 
and height of the bbox. Each label's bbox parameters get normalized 
because the images dimension will be modified/warped to train the 
neural network.
"""

from PIL import Image

data_set_path = "data/train/"
name = "Franklin_Brown"
file_ext = ".labeled_faces.txt"
dest_folder = "labels/"

def create_file(dest_dir: str, file_name: str, content: str):
    file_path = dest_dir + file_name
    with open(file_path, 'w') as file:
        file.write(content)

# Read *.labeled_faces.txt file
with open(data_set_path + name + file_ext, 'r') as file:
    lines = file.readlines()
    for line in lines:
        tokens = line.split(',')
        tokens = [token.strip() for token in tokens]

        ''' BROKEN... sorry
        # Read corresponding image resolution
        img_path = data_set_path + tokens[0].replace("\\", "/")
        img = Image.open(img_path)
        width, height = img.size

        if (width < 300 or height < 300):
            print("file: {} has w/h less than 300 px...".format(img_path))
            break
        '''

        # UNCOMMENT TO NORMALIZE THE LABELS IN THE FILE
        # Create normalized center point, width, and height
        #norm_ctr_x = float(tokens[2]) / width
        #norm_ctr_y = float(tokens[3]) / height
        #norm_bbox_w = float(tokens[4]) / width
        #norm_bbox_h = float(tokens[5]) / height

        # Create string that goes into new txt file
        #label_str = ",".join([tokens[0].replace("\\", "/"), 
        #                      tokens[1],
        #                      str(norm_ctr_x),
        #                      str(norm_ctr_y),
        #                      str(norm_bbox_w),
        #                      str(norm_bbox_h),
        #                      tokens[6],
        #                      tokens[7]])

        label_str = line.replace("\\", "/")

        # Create new txt file
        create_file(data_set_path + dest_folder,
                    tokens[0].replace(".jpg", ".txt").replace("\\", "QsZ"),
                    label_str)
