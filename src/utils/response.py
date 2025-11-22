
from sdks.novavision.src.helper.package import PackageHelper
from components.ImageFilter.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor , OutputImage, OutputText,BasicFilterExecutorResponse,BasicFilterExecutorOutputs,BasicFilter,CompareAndDescribeExecutorResponse,CompareAndDescribeExecutorOutputs,CompareAndDescribe

def build_response_basic_filter(context):
    outputImage = OutputImage(value=context.image)
    BasicFilterExecutorOutputs = BasicFilterExecutorOutputs(outputImage=outputImage)
    BasicFilterExecutorResponse = BasicFilterExecutorResponse(outputs=BasicFilterExecutorOutputs)
    BasicFilter= BasicFilter(value=BasicFilterExecutorResponse)
    executor = ConfigExecutor(value=BasicFilter)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_response_compare_and_describe(context):
    outputImage = OutputImage(value=context.image)
    outputText =OutputText(value=context.text)
    CompareAndDescribeExecutorOutputs = CompareAndDescribeExecutorOutputs(outputImage=outputImage,outputText=outputText)
    CompareAndDescribeExecutorResponse = CompareAndDescribeExecutorResponse(outputs=CompareAndDescribeExecutorOutputs)
    CompareAndDescribe= CompareAndDescribe(value=CompareAndDescribeExecutorResponse)
    executor = ConfigExecutor(value=CompareAndDescribe)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel






