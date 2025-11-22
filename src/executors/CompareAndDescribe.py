"""
    It is one of the preprocessing components in which the image is rotated.
"""

import os
import cv2
import sys
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))


from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.ImageFilterDemo.src.utils.response import build_response_compare_and_describe
from components.ImageFilterDemo.src.models.PackageModel import PackageModel

class CompareAndDescribe(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.image2 = self.request.get_param("inputImage2")
        self.outputFormat = self.request.get_param("outputFormat")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def compare(self, img1, img2):
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        score_ssim, diff = ssim(gray1, gray2, full=True)
        similarity_percentage = score_ssim * 100

        if self.outputFormat == "Percentage":
            return f"{similarity_percentage:.2f}% benzerlik"

        elif self.outputFormat == "TextDescription":
            if similarity_percentage > 90:
                text = "Görüntüler neredeyse tamamen aynı."
            elif similarity_percentage > 70:
                text = "Görüntüler büyük ölçüde benzer."
            elif similarity_percentage > 40:
                text = "Görüntüler kısmen benziyor."
            else:
                text = "Görüntüler oldukça farklı."

            return f"Benzerlik: {similarity_percentage:.2f}% — {text}"

        else:
            return "Geçersiz outputFormat. 'Percentage' veya 'TextDescription' olmalı."

    def run(self):
        img1 = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img2 = Image.get_frame(img=self.image2, redis_db=self.redis_db)
        compare_result = self.compare(img1.value, img2.value)
        packageModel = build_response_compare_and_describe(
            context=self,
            result=compare_result
        )
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
