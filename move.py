"""
move.py
~~~~~~~~~~

A simple script that copies and renames all images in the the revleavent 
dataset_dir subdirectories and puts them into the directory described by
(datset_dir + dest_dir). Note: The images are moved, not copied.
"""
import os

dataset_dir = "data/train/"
dest_dir = "images/"

files_and_dirs = os.listdir(dataset_dir)
list_dirs = [] 

# Remove unwanted files and directories from list of dirs
for file in files_and_dirs:
    if '.' not in file and file != "labels" and file != "images":
        list_dirs.append(file)

# Rename and move images to images folder
for person_dir_name in list_dirs:
    person_path = dataset_dir + person_dir_name + '/'
    for clip_dir_name in os.listdir(person_path):
        clip_path = person_path + clip_dir_name + '/'
        for img_file_name in os.listdir(clip_path):
            img_path = clip_path + img_file_name
            new_img_path = dataset_dir + dest_dir + \
                           img_path.replace(dataset_dir, "").replace("/", "QsZ")

            os.rename(img_path, new_img_path)
