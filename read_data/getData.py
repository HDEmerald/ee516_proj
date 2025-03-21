import os
import cv2

def read_data(file_name, image_path, label_path, image_extension='.png'):
    """
    inputs:
        file_name: name of file without type. Image should be in JPG on PNG format
        image_path: image will be read from this path
        label_path: labels for the image will be read from here

    output:
        image: as a numpy array in rgb format
        label: in [[category_label, x_min, y_min, x_max, y_max]]
    """
    image_path = os.path.join(image_path, file_name + image_extension)
    label_path = os.path.join(label_path, file_name + '.txt')

    assert os.path.exists(image_path), "{0} image not found".format(image_path)
    assert os.path.exists(label_path), "{0} label not found".format(label_path)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # read label
    file = open(label_path)
    read_lines = file.readlines()

    labels = []
    for line in read_lines:
        line = line.strip().split(',')

        c_x = int(line[2])
        c_y = int(line[3])
        w = int(line[4])
        h = int(line[5])
        
        x_min = float(c_x) - 0.5 * float(w)
        y_min = float(c_y) - 0.5 * float(h)
        x_max = float(c_x) + 0.5 * float(w)
        y_max = float(c_y) + 0.5 * float(h)
        
        # 0 is reserved for background
        labels.append([ 1, int(x_min), int(y_min), int(x_max), int(y_max) ])

    return image, labels

def resize_images_and_labels(image, labels, image_height = 300, image_width = 300):
    """
    inputs:
        image: image as numpy array RGB
        label: in [[category_label, x_min, y_min, x_max, y_max]]
        image_height: desired height of image
        image_width: desired width of image
    """

    original_image_height, original_image_width, _ = image.shape

    image = cv2.resize(image, (image_width, image_height))

    labels = [ [    label, 
                    int(x_min * image_width / original_image_width), 
                    int(y_min * image_height / original_image_height), 
                    int(x_max * image_width / original_image_width), 
                    int(y_max * image_height / original_image_height) ] 
                for label, x_min, y_min, x_max, y_max in labels ]

    return image, labels

def label_dimensions_normalized(labels, image_height = 300, image_width = 300):
    """
    input:
        labels: in [[category_label, x_min, y_min, x_max, y_max]]
        image_height: height of image
        image_width: width of image
    output:
        label with dimesion divided by with width and height: in [[category_label, x_min, y_min, x_max, y_max]]
    """
    return [ [ label, x_min / image_width, y_min / image_height, x_max / image_width, y_max /image_height ] 
                for label, x_min, y_min, x_max, y_max in labels ]


