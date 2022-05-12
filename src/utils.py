import PySimpleGUI as sg
import os
import pickle
from pathlib import Path
import cv2


def verify_image(img_file):
    try:
        img = cv2.imread(img_file)
    except:
        return False
    return True


def save_to_pickle(data, filename):
    # save the data to a file
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        print('saved config to file: "{}"'.format(f.name))


def load_from_pickle(filename):
    # load the data from a file
    with open(filename, "rb") as f:
        return pickle.load(f)


def view_dataset_info(folder):
    file_list = os.listdir(folder)
    num_folders = len(file_list)
    # TODO Choice of subset
    image_paths, file_type, classes = load_images_from_folder(folder, subset=None)
    num_images = len(image_paths)
    sg.Popup(
        "Folder: {}\n, Number of images: {}\n, Number of types of images: {}\n, Image formats: {}\n, Classes: {}\n".format(
            folder, num_images, num_folders, file_type, classes
        )
    )


def load_images_from_folder(folder, subset=None):
    image_paths = []
    progwindow = progress_bar(0, len(os.listdir(folder)), "Loading images")
    total_files = 0
    dict_classes = {}
    for index, filename in enumerate(os.listdir(folder)):
        event, values = progwindow.read(timeout=0)
        check_subset = 0
        temp_count = 0
        for image in os.listdir(os.path.join(folder, filename)):
            if subset is not None and check_subset == subset:
                break
            else:
                image_paths.append(os.path.join(folder, filename, image))
                check_subset += 1
                temp_count += 1
        dict_classes[filename] = temp_count
        progwindow["progbar"].update(index)
    progwindow.close()
    file_type = list(set([Path(path).suffix for path in image_paths]))
    file_type = [x for x in file_type if x != ""]
    classes = "\n".join([f"{x}:{dict_classes[x]}" for x in dict_classes.keys()])
    return image_paths, " , ".join(file_type), classes


def bad_image_cleaner(folder, subset=None):
    # TODO Make parallel
    image_paths = []
    progwindow = progress_bar(0, len(os.listdir(folder)), "Verifying images")

    for index, filename in enumerate(os.listdir(folder)):
        event, values = progwindow.read(timeout=0)
        check_subset = 0
        for image in os.listdir(os.path.join(folder, filename)):
            if subset is not None and check_subset == subset:
                break
            else:
                images_suffix = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

                image_path = os.path.join(folder, filename, image)
                if Path(image_path).suffix in images_suffix:
                    if not verify_image(image_path):
                        image_paths.append(image_path)
                check_subset += 1
        progwindow["progbar"].update(index)
    progwindow.close()
    if len(image_paths) > 0:
        sg.Popup(
            "Found {} bad images. Saved them to log file.".format(len(image_paths))
        )
        # TODO What to do with the bad images? Delete? Or Move to a separate folder?
        with open("bad_images.txt", "w") as f:
            f.write("\n".join(image_paths))
    else:
        sg.Popup("No bad images found")
    return image_paths


def progress_bar(progress, max_value, text="Progress Bar"):
    layout = [
        [
            sg.ProgressBar(
                max_value, orientation="h", size=(20, 20), border_width=4, key="progbar"
            )
        ],
    ]
    window = sg.Window(text, layout)
    return window
