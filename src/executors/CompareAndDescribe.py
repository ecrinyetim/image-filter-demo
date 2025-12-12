import os
import cv2
import sys
import numpy as np
import traceback
import base64
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

# GİRİŞ: Görüntü okuma sınıfı
from sdks.novavision.src.media.image import Image
# ÇIKIŞ: Model oluşturma sınıfı (İsim çakışmasını önlemek için ModelImage dedik)
from sdks.novavision.src.base.model import Image as ModelImage

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor

# Helper ve PackageModel
from components.ImageFilterDemo.src.utils.response import build_response_compare_and_describe
from components.ImageFilterDemo.src.models.PackageModel import PackageModel


class CompareAndDescribe(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        # Parametreler
        self.input_image_param = self.request.get_param("inputImage")
        self.input_image2_param = self.request.get_param("inputImage2")
        self.outputFormat = self.request.get_param("outputFormat")
        self.percentage = self.request.get_param("Percentage")
        self.textDesc = self.request.get_param("TextDescription")

        # build_response için değişkenler
        self.image = None
        self.text = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _mse(self, a, b):
        return float(np.mean((a.astype("float32") - b.astype("float32")) ** 2))

    def _ssim(self, imgA, imgB):
        try:
            k1, k2, L = 0.01, 0.03, 255
            C1, C2 = (k1 * L) ** 2, (k2 * L) ** 2
            imgA, imgB = imgA.astype(np.float32), imgB.astype(np.float32)
            kernel, sigma = (11, 11), 1.5
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

    def process(self, img_val1, img_val2):
        """
        Return: (text_result_string, base64_image_string)
        """
        try:
            if img_val1 is None or img_val2 is None:
                return "Error: Images not found", None

            # Numpy array kontrolü
            img = np.array(img_val1) if not isinstance(img_val1, np.ndarray) else img_val1
            img2 = np.array(img_val2) if not isinstance(img_val2, np.ndarray) else img_val2

            # Boyut ve Kanal (3 kanal ise Gri yapma işlemi aşağıda, burda BGR yapıyoruz)
            if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img2.ndim == 2: img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

            # Boyut Eşitleme
            if img.shape[:2] != img2.shape[:2]:
                img2 = cv2.resize(img2, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

            # Gri Tonlama (Hesaplamalar için)
            gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # MSE ve SSIM
            mse_val = self._mse(gray1, gray2)
            mean_ssim = self._ssim(gray1, gray2)
            percentage = mean_ssim * 100.0

            # --- DÜZELTME BAŞLANGICI ---
            # Fark görselini oluştur
            absdiff = cv2.absdiff(gray1, gray2)

            # [WARN FIX] Görüntüyü açıkça uint8 (0-255 tamsayı) formatına çeviriyoruz.
            # SSIM hesaplamalarında float dönüşümleri olduğu için absdiff float kalmış olabilir.
            absdiff = absdiff.astype(np.uint8)

            # Base64 Encode
            b64_string = None
            success, buf = cv2.imencode('.jpg', absdiff)
            if success:
                b64_string = base64.b64encode(buf).decode('utf-8')
            # --- DÜZELTME BİTİŞİ ---

            desc = f"Similarity: {percentage:.2f}%, MSE: {mse_val:.4f}"

            output_val = desc
            req_fmt = str(self.outputFormat).strip() if self.outputFormat else ""
            if req_fmt == "Percentage":
                output_val = f"{percentage:.2f}"

            return str(output_val), b64_string

        except Exception as e:
            # Hata durumunda log basılabilir
            print(f"Process Error: {e}")
            return f"Error: {str(e)}", None

    def run(self):
        # 1. Görüntüleri Redis'ten al
        img_obj1 = Image.get_frame(img=self.input_image_param, redis_db=self.redis_db)
        img_obj2 = Image.get_frame(img=self.input_image2_param, redis_db=self.redis_db)

        val1 = img_obj1.value if img_obj1 else None
        val2 = img_obj2.value if img_obj2 else None

        # 2. İşlemi yap
        text_result, b64_image = self.process(val1, val2)

        # 3. Sonuçları hazırla
        self.text = text_result

        # ModelImage nesnesini oluştur (ImageView bunu okuyacak)
        if b64_image:
            self.image = ModelImage(
                value=b64_image,
                src=f"data:image/jpeg;base64,{b64_image}",
                name="diff_result.jpg",
                type="image",
                uID=str(uuid.uuid4()),
                mimeType="image/jpg",  # ImageView için jpg/png/gif olmalı
                encoding="base64"
            )
        else:
            # ImageView boş veri alırsa çökmemesi için boş ama valid bir yapı dönüyoruz
            self.image = ModelImage(
                value="",
                src="",
                name="error.jpg",
                type="image",
                uID=str(uuid.uuid4()),
                mimeType="image/jpg",
                encoding="base64"
            )

        # 4. Paketi oluştur
        packageModel = build_response_compare_and_describe(context=self)

        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()