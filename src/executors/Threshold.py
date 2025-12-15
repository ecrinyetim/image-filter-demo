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

    def process_images(self, img1, img2):
        try:
            # RESİM 1 İŞLEMİ (Threshold)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            _, t1 = cv2.threshold(gray1, self.thresh_value, 255, cv2.THRESH_BINARY)
            out1 = cv2.cvtColor(t1, cv2.COLOR_GRAY2BGR)

            # RESİM 2 İŞLEMİ (Grayscale)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            out2 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)

            return out1, out2

        except Exception as e:
            traceback.print_exc()
            return img1, img2

    def run(self):
        img_one = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_two = Image.get_frame(img=self.image2, redis_db=self.redis_db)
        res1, res2 = self.process_images(img_one.value, img_two.value)

        self.res_img1 = res1
        self.res_img2 = res2
        packageModel = build_response_threshold(context=self)

        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()