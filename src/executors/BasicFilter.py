"""
    It is one of the preprocessing components in which the image is rotated.
"""
import os
import cv2
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.ImageFilterDemo.src.utils.response import build_response_basic_filter
from components.ImageFilterDemo.src.models.PackageModel import PackageModel

class BasicFilter(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.intensity=self.request.get_param("intensity")
        self.filterType=self.request.get_param("filterType")
        self.image=self.request.get_param("inputImage")


    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def apply_filter(self, img):
        if self.filterType == "Blur":
            ksize = self.intensity * 2 + 1
            return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)

        elif self.filterType == "Sharpen":
            alpha = self.intensity / 10.0
            kernel = np.array([
                [0, -alpha, 0],
                [-alpha, 1 + 4 * alpha, -alpha],
                [0, -alpha, 0]
            ], dtype=np.float32)
            return cv2.filter2D(img, -1, kernel)

        else:
            return img

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img.value = self.apply_filter(img.value)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response_basic_filter(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()