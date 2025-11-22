
from sdks.novavision.src.helper.package import PackageHelper
from components.ImageFilterDemo.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor , OutputImage, OutputText,BasicFilterResponse,BasicFilterOutputs,BasicFilter,CompareAndDescribeResponse,CompareAndDescribeOutputs,CompareAndDescribe

def build_response_basic_filter(context):
    outputImage = OutputImage(value=context.image)
    BasicFilterOutputs = BasicFilterOutputs(outputImage=outputImage)
    BasicFilterResponse = BasicFilterResponse(outputs=BasicFilterOutputs)
    BasicFilter= BasicFilter(value=BasicFilterResponse)
    executor = ConfigExecutor(value=BasicFilter)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_response_compare_and_describe(context):
    outputImage = OutputImage(value=context.image)
    outputText =OutputText(value=context.text)
    CompareAndDescribOutputs = CompareAndDescribeOutputs(outputImage=outputImage,outputText=outputText)
    CompareAndDescribeResponse = CompareAndDescribeResponse(outputs=CompareAndDescribeOutputs)
    CompareAndDescribe= CompareAndDescribe(value=CompareAndDescribeResponse)
    executor = ConfigExecutor(value=CompareAndDescribe)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel






