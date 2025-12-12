"""
    It is one of the preprocessing components in which the image is rotated.
"""

import os
import cv2
import sys
import hashlib

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

        # Sonuç burada saklanır; method override edilmesin diye ayrı attribute kullanalım
        self.compare_result = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def compareAndDesc(self, img,img2):
        import numpy as np
        import cv2
        import base64
        import traceback
        import hashlib

        # Güvenli dönüş helper'ı
        def _safe_return(result_dict):
            # Log: comparison_image_bytes'ı ham basma; bunun yerine len + md5 logla (eğer logger varsa)
            try:
                b = result_dict.get("comparison_image_bytes")
                if isinstance(b, (bytes, bytearray)):
                    length = len(b)
                    md5 = hashlib.md5(b).hexdigest()
                    if hasattr(self, "logger"):
                        try:
                            self.logger.debug("comparison_image_bytes info: len=%d md5=%s", length, md5)
                        except Exception:
                            pass
                else:
                    if hasattr(self, "logger"):
                        try:
                            self.logger.debug("comparison_image_bytes not bytes or missing")
                        except Exception:
                            pass
            except Exception:
                # swallow logging errors
                pass

            # Eğer build_response_compare_and_describe fonksiyonu varsa, önce farklı imzalarla çağırmayı dene
            try:
                if callable(build_response_compare_and_describe):
                    # 1) tek argüman olarak dict ver
                    try:
                        return build_response_compare_and_describe(result_dict)
                    except TypeError:
                        pass
                    # 2) (output, image_bytes)
                    try:
                        return build_response_compare_and_describe(result_dict.get("output"), result_dict.get("comparison_image_bytes"))
                    except TypeError:
                        pass
                    # 3) (output, text_description, image_bytes)
                    try:
                        return build_response_compare_and_describe(result_dict.get("output"), result_dict.get("text_description"), result_dict.get("comparison_image_bytes"))
                    except TypeError:
                        pass
                # Eğer yukarıdakiler çalışmazsa dict'i döndür
            except Exception:
                # build_response... çağrısında hata olursa log ekle fakat yine dict döndür
                try:
                    if hasattr(self, "logger"):
                        self.logger.exception("build_response_compare_and_describe call failed")
                except Exception:
                    pass
            return result_dict

        # --- validation ve dönüşüm ---
        if img is None or img2 is None:
            return _safe_return({
                "error": "Both images must be provided",
                "output": None,
                "comparison_image_bytes": b""
            })

        # Eğer PIL veya başka tip gelmişse numpy array'a çevrilmeye çalış (genelde img numpy array olur)
        # (burada basit kontrol)
        if not isinstance(img, np.ndarray):
            try:
                img = np.array(img)
            except Exception:
                return _safe_return({
                    "error": "img is not a numpy array and cannot be converted",
                    "output": None,
                    "comparison_image_bytes": b""
                })
        if not isinstance(img2, np.ndarray):
            try:
                img2 = np.array(img2)
            except Exception:
                return _safe_return({
                    "error": "img2 is not a numpy array and cannot be converted",
                    "output": None,
                    "comparison_image_bytes": b""
                })

        # Eğer tek kanallı gelirse BGR'ye çevir (görsel oluşturmak için)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img2.ndim == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # Boyutları eşitle (img2'yi img boyutuna getir)
        try:
            if img.shape[:2] != img2.shape[:2]:
                img2 = cv2.resize(img2, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        except Exception:
            # Eğer resize başarısız olursa hata döndür
            return _safe_return({
                "error": "Failed to resize images to same dimensions",
                "output": None,
                "comparison_image_bytes": b""
            })

        # --- metric helper'ları ---
        def _to_gray(i):
            return cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i

        def _mse(a, b):
            a_f = a.astype("float32")
            b_f = b.astype("float32")
            return float(np.mean((a_f - b_f) ** 2))

        def _ssim(imgA, imgB, k1=0.01, k2=0.03, L=255):
            # single-channel SSIM implementation (özet)
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
            # farkları eşikle ve konturları bul
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

        # --- hesaplamalar ---
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

            # açıklama metni
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

            # görseli jpeg bytes'a çevir
            success, buf = cv2.imencode('.jpg', comp_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            comparison_image_bytes = buf.tobytes() if success else b""

            # outputFormat kontrolü (bazı input'lar boolean/None olabilir)
            out_fmt = self.outputFormat if isinstance(self.outputFormat, str) else (str(self.outputFormat) if self.outputFormat is not None else "")
            if isinstance(out_fmt, str) and out_fmt.lower() == "percentage":
                output_value = round(percentage, 2)
            else:
                output_value = text_description

            result = {
                "output": output_value,
                "percentage": round(percentage, 2),
                "text_description": text_description,
                "comparison_image_bytes": comparison_image_bytes,
                "num_diff_regions": num_diff_regions,
                "diff_area_percent": diff_area_percent,
                "mse": mse_val
            }

            return _safe_return(result)

        except Exception as ex:
            # hata durumunda traceback ile dict dön
            tb = traceback.format_exc()
            return _safe_return({
                "error": "Exception during comparison",
                "exception": str(ex),
                "traceback": tb
            })


    def run(self):
        # DÜZELTME: img2'yi self.image2 ile al
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img2 = Image.get_frame(img=self.image2, redis_db=self.redis_db)
        self.compareAndDesc = self.compareAndDesc(img.value, img2.value)
        if isinstance(self.compareAndDesc , dict): self.compareAndDesc  = [self.compareAndDesc]
        PackageModel = build_response_compare_and_describe(context=self)
        return PackageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
