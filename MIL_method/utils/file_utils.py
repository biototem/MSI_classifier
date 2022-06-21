import os


IMG_EXTENSIONS = ['.svs', '.npdi']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def getAllImageName(img_dir):
    img_name = []
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            if has_file_allowed_extension(f, IMG_EXTENSIONS):
                img_name.append(".".join(f.split(".")[:-1]))
    return img_name


def getAllImagePath(img_dir):
    img_path = []
    if not isinstance(img_dir, list):
        img_dir = [img_dir]
    for path in img_dir:
        for root, dirs, files in os.walk(path):
            for f in files:
                if has_file_allowed_extension(f, IMG_EXTENSIONS):
                    img_path.append(os.path.join(root, f))
    return img_path