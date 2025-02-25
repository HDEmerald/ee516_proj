from PIL import Image
import csv

"""
compress.py
~~~~~~~~~~~

A module that compresses and crops the data in a dataset to make it
readable to the model that will be trained and evaluated by it.

Use with caution!!!

(https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset)
"""

def process_archive(archive_loc: str, csv_loc: str, archive_dest: str):
    # Open/read the archive's .csv file
    with open(csv_loc, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if (row[0] != "image"):
                if (is_num_capital(row[1])):
                    image = process_im(archive_loc + "/" + row[0])
                    image.save(archive_dest + "/" + row[0])


def is_num_capital(char):
    # return True if the data belongs in the new processed dataset
    return '0' <= char <= '9' or 'A' <= char <= 'N' or 'P' <= char <= 'Z'

def process_im(file_loc: str):
    # Open the image
    image = Image.open(file_loc)
    width, height = image.size

    # Crop the image
    crop = int((width - height) / 2)
    image = image.crop((crop, 0, height + crop, height))

    # Compress the image
    size = (19, 19)
    image.thumbnail(size)

    return image

process_archive("archive", "archive/english.csv", "c_archive")