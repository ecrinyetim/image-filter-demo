import os
import cv2
import sys
import numpy as np
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor

# Referans koddaki gibi utils ve models importları
from components.ImageFilterDemo.src.utils.response import build_response_compare_and_describe
from components.ImageFilterDemo.src.models.PackageModel import PackageModel

class CompareAndDescribe(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        # Referanstaki gibi Model initialize ediliyor
        self.request.model = PackageModel(**(self.request.data))
        
        # Parametrelerin alınması
        self.image = self.request.get_param("inputImage")
        self.image2 = self.request.get_param("inputImage2")
        self.outputFormat = self.request.get_param("outputFormat")
        self.percentage = self.request.get_param("Percentage")
        self.textDesc = self.request.get_param("TextDescription")

        # Sonuçları tutacak değişken (Referanstaki self.verify mantığına karşılık gelir)
        self.compare_result = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _mse(self, a, b):
        # Yardımcı fonksiyonları metodun içine gömmek yerine class metodu yapmak daha temizdir
        return float(np.mean((a.astype("float32") - b.astype("float32")) ** 2))

    def _ssim(self, imgA, imgB):
        # Basitleştirilmiş SSIM mantığı
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
            sigma1_sq = cv2.GaussianBlur(imgA**2, kernel, sigma) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(imgB**2, kernel, sigma) - mu2_sq
            sigma12 = cv2.GaussianBlur(imgA * imgB, kernel, sigma) - mu1_mu2
            num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            ssim_map = num / den
            return float(np.mean(ssim_map))
        except Exception:
            return 0.0

    def compareAndDesc(self, img, img2):
        """
        Bu metod referans koddaki 'verify' metodu gibi çalışır.
        Sadece hesaplama yapar ve Dict (veya hata durumunda List) döner.
        """
        try:
            # Validasyonlar
            if img is None or img2 is None:
                return [{"error": "Both images must be provided", "verified": False}]

            # Numpy array kontrolü
            if not isinstance(img, np.ndarray): img = np.array(img)
            if not isinstance(img2, np.ndarray): img2 = np.array(img2)

            # Griye çevirme
            if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img2.ndim == 2: img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

            # Resize (img2 -> img boyutuna)
            if img.shape[:2] != img2.shape[:2]:
                img2 = cv2.resize(img2, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

            # Hesaplamalar
            gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            mse_val = self._mse(gray1, gray2)
            mean_ssim = self._ssim(gray1, gray2)
            percentage = mean_ssim * 100.0

            # Görsel fark oluşturma (Bytes)
            absdiff = cv2.absdiff(gray1, gray2)
            success, buf = cv2.imencode('.jpg', absdiff)
            comparison_bytes = buf.tobytes() if success else b""

            desc = f"Similarity: {percentage:.2f}%, MSE: {mse_val:.4f}"
            
            # Output format belirleme
            output_value = desc
            if self.outputFormat and str(self.outputFormat).lower() == "percentage":
                output_value = round(percentage, 2)

            # Tıpkı referans kodun 'resp_obj' döndürmesi gibi bir sözlük döndürüyoruz
            resp_obj = {
                "output": output_value,
                "percentage": round(percentage, 2),
                "text_description": desc,
                "comparison_image_bytes": comparison_bytes,
                "mse": mse_val,
                "verified": True # Standart olması için eklenebilir
            }
            return resp_obj

        except Exception as e:
            # Referanstaki hata yakalama stili
            return [{"verified": False, "error": str(e), "trace": traceback.format_exc()}]

    def run(self):
        # 1. Görüntüleri al (Referans koddaki yapı)
        img_obj1 = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img_obj2 = Image.get_frame(img=self.image2, redis_db=self.redis_db)

        # Görüntülerin gelip gelmediğini kontrol edelim (get_frame None dönebilir)
        val1 = img_obj1.value if img_obj1 else None
        val2 = img_obj2.value if img_obj2 else None

        # 2. Logic'i çalıştır ve sonucu instance değişkenine ata
        # Referans kodda: self.verify = self.verify(...) yapılmıştı.
        self.compare_result = self.compareAndDesc(val1, val2)

        # 3. Sonucu List formatına çevir (Referanstaki yapı: if isinstance(...) list wrap)
        if isinstance(self.compare_result, dict): 
            self.compare_result = [self.compare_result]

        # 4. Response Builder'ı çağır
        # BURASI ÇOK ÖNEMLİ: Referans kodda parametre olarak 'context=self' veriliyor.
        # Bu sayede helper fonksiyon (utils/response.py), 'self.compare_result' verisine erişebilir.
        packageModel = build_response_compare_and_describe(context=self)

        # 5. Modeli döndür
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()