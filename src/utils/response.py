
from sdks.novavision.src.helper.package import PackageHelper
from components.ImageFilterDemo.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor , OutputImage, OutputText, BasicFilterResponse, BasicFilterOutputs, BasicFilter, CompareAndDescribeResponse, CompareAndDescribeOutputs, CompareAndDescribe

def build_response_basic_filter(context):
    outputImage = OutputImage(value=context.image)
    basicFilterOutputs = BasicFilterOutputs(outputImage=outputImage)
    basicFilterResponse = BasicFilterResponse(outputs=BasicFilterOutputs)
    basicFilter= BasicFilter(value=BasicFilterResponse)
    executor = ConfigExecutor(value=BasicFilter)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_compare_and_describe(context):
    outputImage = OutputImage(value=context.image)
    outputText =OutputText(value=context.text)
    compareAndDescribOutputs = CompareAndDescribeOutputs(outputImage=outputImage,outputText=outputText)
    compareAndDescribeResponse = CompareAndDescribeResponse(outputs=CompareAndDescribeOutputs)
    compareAndDescribe= CompareAndDescribe(value=CompareAndDescribeResponse)
    executor = ConfigExecutor(value=CompareAndDescribe)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel






