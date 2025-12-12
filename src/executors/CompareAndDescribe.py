"""
    It is one of the preprocessing components in which the image is rotated.
"""

import os
import cv2
import sys
import numpy as np
import traceback
import base64  # Base64 importu en tepeye eklendi

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
        self.percentage = self.request.get_param("Percentage")
        self.textDesc = self.request.get_param("TextDescription")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def compareAndDesc(self, img, img2):

        import numpy as np
        import cv2
        import traceback
        import base64  # <--- 1. BURAYA EKLENDİ

        # --- validation ---
        # Hata durumunda bytes (b"") yerine boş string ("") dönmelisiniz.
        if img is None or img2 is None:
            return {"error": "Both images must be provided", "output": None, "comparison_image_bytes": ""}

        # ... (Kodun bu kısımları aynı kalabilir, array kontrolü vs.) ...
        if not isinstance(img, np.ndarray):
            try:
                img = np.array(img)
            except Exception:
                return {"error": "img is not a numpy array", "output": None, "comparison_image_bytes": ""}
        if not isinstance(img2, np.ndarray):
            try:
                img2 = np.array(img2)
            except Exception:
                return {"error": "img2 is not a numpy array", "output": None, "comparison_image_bytes": ""}

        # Ensure BGR 3-channel
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img2.ndim == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # Resize img2 to img if needed
        try:
            if img.shape[:2] != img2.shape[:2]:
                img2 = cv2.resize(img2, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        except Exception:
            return {"error": "Failed to resize images", "output": None, "comparison_image_bytes": ""}

        # helpers
        def _to_gray(i):
            return cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i

        def _mse(a, b):
            a_f = a.astype("float32")
            b_f = b.astype("float32")
            return float(np.mean((a_f - b_f) ** 2))

        def _ssim(imgA, imgB, k1=0.01, k2=0.03, L=255):
            # ... (SSIM fonksiyonu aynı kalacak) ...
            C1 = (k1 * L) ** 2
            C2 = (k2 * L) ** 2
            imgA_f = imgA.astype(np.float32)
            imgB_f = imgB.astype(np.float32)
            kernel = (11, 11)
            sigma = 1.5
            mu1 = cv2.GaussianBlur(imgA_f, kernel, sigma)
            mu2 = cv2.GaussianBlur(imgB_f, kernel, sigma)
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.GaussianBlur(imgA_f * imgA_f, kernel, sigma) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(imgB_f * imgB_f, kernel, sigma) - mu2_sq
            sigma12 = cv2.GaussianBlur(imgA_f * imgB_f, kernel, sigma) - mu1_mu2
            num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            ssim_map = np.ones_like(num, dtype=np.float32)
            mask = den != 0
            ssim_map[mask] = (num[mask] / den[mask])
            ssim_map = np.clip(ssim_map, -1.0, 1.0)
            mean_ssim = float(np.mean(ssim_map))
            mean_ssim = max(0.0, min(1.0, mean_ssim))
            return mean_ssim, ssim_map

        def _make_vis(a_bgr, b_bgr, diff_gray, thresh=30):
            # ... (Görselleştirme fonksiyonu aynı kalacak) ...
            _, th = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
            th = cv2.morphologyEx(th, cv2.MORPH_DILATE, kernel, iterations=1)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay = b_bgr.copy()
            for c in contours:
                if cv2.contourArea(c) < 10:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                sub = overlay[y:y + h, x:x + w]
                color = np.array([0, 0, 255], dtype=np.uint8)
                alpha = 0.25
                overlay[y:y + h, x:x + w] = cv2.addWeighted(sub, 1 - alpha, np.full_like(sub, color), alpha, 0)
            norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            h, w = a_bgr.shape[:2]
            if overlay.shape[:2] != (h, w):
                overlay = cv2.resize(overlay, (w, h))
                heat = cv2.resize(heat, (w, h))
            combined = np.concatenate([a_bgr, overlay, heat], axis=1)
            return combined, contours, th

        # compute metrics and visualization
        try:
            gray1 = _to_gray(img)
            gray2 = _to_gray(img2)
            mse_val = _mse(gray1, gray2)
            mean_ssim, ssim_map = _ssim(gray1, gray2)
            percentage = mean_ssim * 100.0
            absdiff = cv2.absdiff(gray1, gray2)
            comp_vis, contours, th = _make_vis(img, img2, absdiff, thresh=30)

            num_diff_regions = sum(1 for c in contours if cv2.contourArea(c) >= 10)
            total_diff_pixels = int(np.count_nonzero(th))
            total_pixels = th.shape[0] * th.shape[1] if th is not None else 1
            diff_area_percent = (total_diff_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

            # description
            desc = []
            desc.append(f"Similarity (SSIM-based): {percentage:.2f}%")
            desc.append(f"MSE: {mse_val:.4f}")
            desc.append(f"Number of differing regions (area >= 10 px): {num_diff_regions}")
            desc.append(f"Total differing pixels: {total_diff_pixels} ({diff_area_percent:.4f}%)")
            if percentage > 98.0 and diff_area_percent < 0.1:
                desc.append("Yorum: Görseller neredeyse aynı (çok küçük farklılıklar).")
            elif percentage > 90.0:
                desc.append("Yorum: Görseller yüksek oranda benzer; bazı küçük farklılıklar var.")
            elif percentage > 70.0:
                desc.append("Yorum: Görseller kısmen benzer; farklar belirgin.")
            else:
                desc.append("Yorum: Görseller büyük ölçüde farklı.")
            text_description = "\n".join(desc)

            # ---------------------------------------------------------
            # 2. DÜZELTME BURADA: Görseli Base64 String'e çeviriyoruz
            # ---------------------------------------------------------

            # encode visualization to jpeg bytes
            success, buf = cv2.imencode('.jpg', comp_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            comparison_image_bytes = ""
            if success:
                # Raw bytes yerine Base64 string kullanıyoruz.
                # .decode('utf-8') ile bytes objesini string'e çeviriyoruz.
                comparison_image_bytes = base64.b64encode(buf).decode('utf-8')

            # prepare plain dict result
            out_fmt = self.outputFormat if isinstance(self.outputFormat, str) else (
                str(self.outputFormat) if self.outputFormat is not None else "")
            if isinstance(out_fmt, str) and out_fmt.lower() == "percentage":
                output_value = round(percentage, 2)
            else:
                output_value = text_description

            result = {
                "output": output_value,
                "percentage": round(percentage, 2),
                "text_description": text_description,
                "comparison_image_bytes": comparison_image_bytes,  # Artık string formatında
                "num_diff_regions": num_diff_regions,
                "diff_area_percent": diff_area_percent,
                "mse": mse_val
            }

            return result

        except Exception as ex:
            tb = traceback.format_exc()
            # Hata durumunda da boş string dönmeli
            return {"error": "Exception during comparison", "exception": str(ex), "traceback": tb,
                    "comparison_image_bytes": ""}


    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img2 = Image.get_frame(img=self.image2, redis_db=self.redis_db)

        self.compareAndDesc = self.compareAndDesc(img.value, img2.value)

        if isinstance(self.compareAndDesc, dict):
            self.compareAndDesc = [self.compareAndDesc]

        packageModel = build_response_compare_and_describe(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()