import cv2

class ApplyColorMap(object):
    def __init__(self) -> None:
        pass

    def __call__(self, image):
        return cv2.applyColorMap(image, cv2.COLORMAP_PARULA)