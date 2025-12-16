import os
import cv2
import sys
import numpy as np
import traceback
import base64
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.model import Image as ModelImage
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.ImageFilterDemo.src.utils.response import build_response_threshold
from components.ImageFilterDemo.src.models.PackageModel import PackageModel


class Threshold(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.image2 = self.request.get_param("inputImage2")

        self.thresh_value = self.request.get_param("thresholdValue")
        if self.thresh_value is None:
            self.thresh_value = 127
        else:
            self.thresh_value = int(self.thresh_value)

        self.res_img1 = None
        self.res_img2 = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def apply_threshold(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, self.thresh_value, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        except Exception:
            traceback.print_exc()
            return img

    # --- METHOD 2: Sadece Grayscale İşlemi ---
    def apply_grayscale(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception:
            traceback.print_exc()
            return img

    def run(self):
        img_one = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_two = Image.get_frame(img=self.image2, redis_db=self.redis_db)
        img_one.value= self.apply_threshold(img_one.value)
        img_two.value = self.apply_grayscale(img_two.value)
        self.image = Image.set_frame(img=img_one,package_uID=self.uID, redis_db=self.redis_db)
        self.image2 = Image.set_frame(img=img_two,package_uID=self.uID, redis_db=self.redis_db)

        packageModel = build_response_threshold(context=self)

        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()