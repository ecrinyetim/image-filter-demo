
from sdks.novavision.src.helper.package import PackageHelper
from components.Package.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor , OutputImage, OutputText,BasicFilterExecutorResponse,BasicFilterExecutorOutputs,BasicFilterExecutor,CompareAndDescribeExecutorResponse,CompareAndDescribeExecutorOutputs,CompareAndDescribeExecutor


def build_response_basic_filter(context):
    outputImage = OutputImage(value=context.image)
    BasicFilterExecutorOutputs = BasicFilterExecutorOutputs(outputImage=outputImage)
    BasicFilterExecutorResponse = BasicFilterExecutorResponse(outputs=BasicFilterExecutorOutputs)
    BasicFilterExecutor = BasicFilterExecutor(value=BasicFilterExecutorResponse)
    executor = ConfigExecutor(value=BasicFilterExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_compare_and_describe(context):
    outputImage = OutputImage(value=context.image)
    outputText =OutpuText(value=context.text)
    CompareAndDescribeExecutorOutputs = CompareAndDescribeExecutorOutputs(outputImage=outputImage,outputText=outputText)
    CompareAndDescribeExecutorResponse = CompareAndDescribeExecutorResponse(outputs=CompareAndDescribeExecutorOutputs)
    CompareAndDescribeExecutor = CompareAndDescribeExecutor(value=CompareAndDescribeExecutorResponse)
    executor = ConfigExecutor(value=CompareAndDescribeExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel






