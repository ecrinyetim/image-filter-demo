import os
import cv2
import sys
import numpy as np
import traceback
import base64  # Base64 eklendi

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor

# Utils ve Models importları
from components.ImageFilterDemo.src.utils.response import build_response_compare_and_describe
from components.ImageFilterDemo.src.models.PackageModel import PackageModel


class CompareAndDescribe(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.image2 = self.request.get_param("inputImage2")

        # Parametreleri alırken string kontrolü yapalım
        self.outputFormat = self.request.get_param("outputFormat")
        self.percentage = self.request.get_param("Percentage")
        self.textDesc = self.request.get_param("TextDescription")

        # Sonucu saklayacağımız değişken
        self.compare_result = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _mse(self, a, b):
        return float(np.mean((a.astype("float32") - b.astype("float32")) ** 2))

    def _ssim(self, imgA, imgB):
        try:
            k1, k2, L = 0.01, 0.03, 255
            C1 = (k1 * L) ** 2
            C2 = (k2 * L) ** 2
            imgA = imgA.astype(np.float32)
            imgB = imgB.astype(np.float32)
            kernel = (11, 11)
            sigma = 1.5
            mu1 = cv2.GaussianBlur(imgA, kernel, sigma)
            mu2 = cv2.GaussianBlur(imgB, kernel, sigma)
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.GaussianBlur(imgA ** 2, kernel, sigma) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(imgB ** 2, kernel, sigma) - mu2_sq
            sigma12 = cv2.GaussianBlur(imgA * imgB, kernel, sigma) - mu1_mu2
            num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            ssim_map = num / den
            return float(np.mean(ssim_map))
        except Exception:
            return 0.0

    def compareAndDesc(self, img, img2):
        try:
            if img is None or img2 is None:
                return {"error": "Both images must be provided", "verified": False}

            if not isinstance(img, np.ndarray): img = np.array(img)
            if not isinstance(img2, np.ndarray): img2 = np.array(img2)

            if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img2.ndim == 2: img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

            if img.shape[:2] != img2.shape[:2]:
                img2 = cv2.resize(img2, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

            gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            mse_val = self._mse(gray1, gray2)
            mean_ssim = self._ssim(gray1, gray2)
            percentage = mean_ssim * 100.0

            # Görsel farkı oluşturma
            absdiff = cv2.absdiff(gray1, gray2)
            success, buf = cv2.imencode('.jpg', absdiff)

            # --- DÜZELTME BURADA BAŞLIYOR ---
            # Hatayı önlemek için Bytes -> Base64 String dönüşümü yapıyoruz.
            comparison_b64 = ""
            if success:
                comparison_b64 = base64.b64encode(buf).decode('utf-8')
            # --------------------------------

            desc = f"Similarity: {percentage:.2f}%, MSE: {mse_val:.4f}"

            # --- Output Format Mantığı ---
            # Kullanıcının outputFormat isteğine göre çıktı belirleme
            output_value = desc  # Varsayılan (TextDescription mantığı)

            requested_fmt = str(self.outputFormat).strip() if self.outputFormat else ""

            if requested_fmt == "Percentage":
                output_value = round(percentage, 2)
            elif requested_fmt == "TextDescription":
                output_value = desc
            # -----------------------------

            resp_obj = {
                "output": output_value,
                "percentage": round(percentage, 2),
                "text_description": desc,
                "comparison_image_bytes": comparison_b64,  # String olarak gönderiyoruz
                "mse": mse_val,
                "verified": True
            }
            return resp_obj

        except Exception as e:
            # Hata durumunda traceback'i string olarak dönüyoruz, raw bytes hatası almamak için
            return {"verified": False, "error": str(e), "trace": str(traceback.format_exc())}

    def run(self):
        img_obj1 = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_obj2 = Image.get_frame(img=self.image2, redis_db=self.redis_db)

        val1 = img_obj1.value if img_obj1 else None
        val2 = img_obj2.value if img_obj2 else None

        # Sonucu hesapla
        self.compare_result = self.compareAndDesc(val1, val2)

        # Verify örneğindeki gibi, eğer sonuç bir dict ise listeye sarıyoruz.
        # Bu, build_response fonksiyonunun iterate etmesini (döngüye girmesini) beklediğini gösteriyor.
        if isinstance(self.compare_result, dict):
            self.compare_result = [self.compare_result]

        # Context (self) göndererek yanıtı oluşturuyoruz
        packageModel = build_response_compare_and_describe(context=self)

        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()