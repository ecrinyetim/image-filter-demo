"""
    It is one of the preprocessing components in which the image is rotated.
"""

import os
import cv2
import sys
import tensorflow as tf
import tensorflow_addons as tfa


sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.Package.src.utils.response import build_response_basic_filter
from components.Package.src.models.PackageModel import PackageModel

class BasicFilter(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.intensity=self.request.get_param("intensity")
        self.filterType=self.request.get_param("filterType")
        self.image=self.request.get_param("inputImage")


    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def blur(self, img):
        sigma=self.intensity
        kernel_size = int(self.intensity * 2 + 1)
        kernel_shape = (kernel_size, kernel_size)

        blurred = tfa.image.gaussian_filter2d(
            image=img,
            filter_shape=kernel_shape,
            sigma=sigma,
            padding="REFLECT"
        )
        return blurred

    def sharpen(self,img):
        k = -1.0 * tf.ones((3, 3), dtype=tf.float32)
        k = tf.Variable(k)
        k[1, 1].assign(1.0 + self.intensity)
        channels = img.shape[-1] if len(img.shape) == 3 else 1
        kernel = tf.repeat(k[:, :, tf.newaxis, tf.newaxis], repeats=channels, axis=2)

        if len(img.shape) == 3:
            img = img[tf.newaxis, ...]

        sharpened = tf.nn.conv2d(img, kernel, strides=1, padding="SAME")

        if sharpened.shape[0] == 1:
            sharpened = sharpened[0]

        sharpened = tf.clip_by_value(sharpened, 0.0, 1.0)

        return sharpened

    def run(self):
        img=Image.get_frame(img=self.image,redis_db=self.redis_db)
        if(self.filterType=="Blur"):
            img.value=self.blur(img.value)
        else:
            img.value=self.sharpen(img.value)

        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response_basic_filter(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()