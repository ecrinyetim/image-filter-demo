
from sdks.novavision.src.helper.package import PackageHelper
from components.ImageFilterDemo.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor , OutputImage, OutputImage2, BasicFilterResponse, BasicFilterOutputs, BasicFilter, ThresholdResponse, ThresholdOutputs, Threshold

def build_response_basic_filter(context):
    outputImage = OutputImage(value=context.image)
    basicFilterOutputs = BasicFilterOutputs(outputImage=outputImage)
    basicFilterResponse = BasicFilterResponse(outputs=basicFilterOutputs)
    basicFilter= BasicFilter(value=basicFilterResponse)
    executor = ConfigExecutor(value=basicFilter)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_response_threshold(context):
    outputImage= OutputImage(value=context.image)
    outputImage2= OutputText(value=context.image)
    thresholdOutputs = ThresholdOutputs(outputImage=outputImage, outputText=outputText)
    thresholdResponse = ThresholdResponse(outputs=thresholdOutputs)
    threshold = Threshold(value=thresholdResponse)
    executor= ConfigExecutor(value=threshold)
    package_configs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    packageModel = package.build_model(context)

    return packageModel





