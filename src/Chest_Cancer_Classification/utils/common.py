import os 
import json
from tkinter import Image
import joblib
import numpy as np
import yaml
import base64
from pathlib import Path
from typing import Any, Union
from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from Chest_Cancer_Classification import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        raise ValueError("yaml file is empty") from e
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    
@ensure_annotations
def resize_image(image_path: Path, output_size: tuple) -> np.ndarray:
    """Resize an image to a specified output size.

    Args:
        image_path (Path): Path to the image file.
        output_size (tuple): Desired output size in the format (width, height).

    Returns:
        np.ndarray: Resized image as a NumPy array.
    """
    img = Image.open(image_path)
    img = img.resize(output_size, Image.ANTIALIAS)
    return np.array(img)


@ensure_annotations
def convert_image_to_array(image_path: Path) -> np.ndarray:
    """Convert an image to a NumPy array.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        np.ndarray: Image as a NumPy array.
    """
    img = Image.open(image_path)
    return np.array(img)


@ensure_annotations
def save_image_from_array(image_array: np.ndarray, output_path: Path):
    """Save an image from a NumPy array to a specified path.

    Args:
        image_array (np.ndarray): Image as a NumPy array.
        output_path (Path): Path to save the image.
    """
    img = Image.fromarray(image_array)
    img.save(output_path)


@ensure_annotations
def plot_image(image_array: np.ndarray):
    """Plot an image from a NumPy array.

    Args:
        image_array (np.ndarray): Image as a NumPy array.
    """
    img = Image.fromarray(image_array)
    img.show()


@ensure_annotations
def crop_image(image_array: np.ndarray, coordinates: tuple) -> np.ndarray:
    """Crop an image based on specified coordinates.

    Args:
        image_array (np.ndarray): Image as a NumPy array.
        coordinates (tuple): Coordinates for cropping in the format (x_start, y_start, x_end, y_end).

    Returns:
        np.ndarray: Cropped image as a NumPy array.
    """
    x_start, y_start, x_end, y_end = coordinates
    cropped_img = image_array[y_start:y_end, x_start:x_end]
    return cropped_img


@ensure_annotations
def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Merge two dictionaries.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        dict: Merged dictionary.
    """
    return {**dict1, **dict2}


@ensure_annotations
def check_data_type(data: Union[str, int, float, list, dict, tuple]) -> str:
    """Check the type of data provided.

    Args:
        data (Union[str, int, float, list, dict, tuple]): Input data of various types.

    Returns:
        str: String representation of the data type.
    """
    return str(type(data))


@ensure_annotations
def remove_duplicates_from_list(input_list: list) -> list:
    """Remove duplicates from a list while preserving the order of elements.

    Args:
        input_list (list): Input list.

    Returns:
        list: List with duplicates removed.
    """
    return list(dict.fromkeys(input_list))
