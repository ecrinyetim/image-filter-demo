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

        self.method = self.request.get_param("configCalculationMethod")  # Örn: 'Fixed' veya 'Auto'

        # Eğer 'Fixed' seçildiyse slider değerini al, yoksa varsayılan 127 olsun
        self.thresh_value = self.request.get_param("configThresholdValue")
        if self.thresh_value is None:
            self.thresh_value = 127
        else:
            self.thresh_value = int(self.thresh_value)

        # Çıktıları saklayacağımız değişkenler
        self.output_image = None  # Birleştirilmiş resim için
        self.output_text = None  # Rapor metni için


    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}


    def thresholding(self, img1, img2):
        try:
            # 1. Resimleri Gri Tonlamaya Çevir (Threshold için şart)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # 2. Threshold İşlemi Uygula
            _, t1 = cv2.threshold(gray1, self.thresh_value, 255, cv2.THRESH_BINARY)
            _, t2 = cv2.threshold(gray2, self.thresh_value, 255, cv2.THRESH_BINARY)

            # 3. Resimleri Yan Yana Birleştirme (Hconcat)
            h1, w1 = t1.shape
            h2, w2 = t2.shape

            if h1 != h2:
                scale = h1 / h2
                new_w = int(w2 * scale)
                t2 = cv2.resize(t2, (new_w, h1))

            combined = cv2.hconcat([t1, t2])

            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

            # 4. Rapor Metnini Oluştur
            info_text = f"Resim 1 Threshold Degeri: {self.thresh_value} | Resim 2 Threshold Degeri: {self.thresh_value}"

            return combined_bgr, info_text

        except Exception as e:
            traceback.print_exc()
            return None, f"Hata olustu: {str(e)}"


    def run(self):
        img_one = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_two = Image.get_frame(img=self.image2, redis_db=self.redis_db)

        processed_img, info_text = self.thresholding(img_one.value, img_two.value)

        self.output_image = processed_img
        self.output_text = info_text

        packageModel = build_response_threshold(context=self)

        return packageModel


    if "__main__" == __name__:
        Executor(sys.argv[1]).run()