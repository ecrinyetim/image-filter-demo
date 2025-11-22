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

        score, diff = ssim(gray1, gray2, full=True)
        similarity_percentage = score * 100

        diff = (diff * 255).astype("uint8")
        diff_color = cv2.applyColorMap(255 - diff, cv2.COLORMAP_JET)

        if self.outputFormat == "Percentage":
            text = f"{similarity_percentage:.2f}% benzerlik"
        else:
            if similarity_percentage > 90:
                desc = "Görüntüler neredeyse tamamen aynı."
            elif similarity_percentage > 70:
                desc = "Görüntüler büyük ölçüde benzer."
            elif similarity_percentage > 40:
                desc = "Görüntüler kısmen benziyor."
            else:
                desc = "Görüntüler oldukça farklı."
            text = f"Benzerlik: {similarity_percentage:.2f}% — {desc}"

        return text, diff_color

    def run(self):
        img1 = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img2 = Image.get_frame(img=self.image2, redis_db=self.redis_db)

        compare_text, diff_image = self.compare(img1.value, img2.value)

        diff_img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        diff_img.value = diff_image

        self.compareImage = Image.set_frame(
            img=diff_img,
            package_uID=self.uID,
            redis_db=self.redis_db
        )

        self.compareResult = compare_text
        packageModel = build_response_compare_and_describe(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()

