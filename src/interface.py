import PySimpleGUI as sg
import os.path
import json
from .utils import *
import os
import cv2
from pathlib import Path


def main_window():

    file_list_column = [
        [
            [sg.Text("What do you want to do?")],
            [sg.Button("Choose a folder")],
            [sg.Button("View dataset information")],
            [sg.Button("Bad Image Cleaner")],
            [sg.Button("Train a model")],
            [sg.Button("Quit")],
        ],
    ]

    window = sg.Window("Reference Sorter", file_list_column)

    if os.path.exists("data.pkl"):
        data = load_from_pickle("data.pkl")
        folder_chosen = data["folder"]
        data_verified = data["verified"]
    else:
        folder_chosen = ""
        data_verified = False
        data = {"folder": "", "verified": False}

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED or event == "Quit":
            break
        elif event == "Choose a folder":
            if len(folder_chosen) == 0:
                folder_chosen = create_folder_picker()
                data["folder"] = folder_chosen
            else:
                if (
                    sg.PopupYesNo(
                        f"Your current folder is {data['folder']}\n Do you want to change the folder?"
                    )
                    == "Yes"
                ):
                    folder_chosen = create_folder_picker()
                    data["folder"] = folder_chosen
                    data_verified = False
                    data["verified"] = False

        elif event == "View dataset information":
            if len(folder_chosen) == 0:
                sg.Popup("You need to choose a folder first")
            else:
                view_dataset_info(folder_chosen)
        elif event == "Bad Image Cleaner":
            if len(folder_chosen) == 0:
                sg.Popup("You need to choose a folder first")
            else:
                if data_verified is True:
                    if (
                        sg.PopupYesNo(
                            "You already verified the images. Do you want to reverify?"
                        )
                        == "Yes"
                    ):
                        bad_images = bad_image_cleaner(folder_chosen, subset=None)
                else:
                    if (
                        sg.PopupYesNo(
                            "Are you sure you want to verify the dataset? It will take time."
                        )
                        == "Yes"
                    ):
                        bad_images = bad_image_cleaner(folder_chosen, subset=None)
                        data_verified = True
                        data["verified"] = True
    window.close()

    save_to_pickle(data, "data.pkl")


def create_folder_picker():
    file_list_column = [
        [
            sg.Text("Pick the main folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
            sg.Button("OK"),
        ],
    ]
    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
        ]
    ]

    window = sg.Window("Image Viewer", layout)
    folder_chosen = ""

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED or event == "OK":
            break
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            folder_chosen = folder
    window.close()
    return folder_chosen
