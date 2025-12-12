
from sdks.novavision.src.helper.package import PackageHelper
from components.ImageFilterDemo.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor , OutputImage, OutputText, BasicFilterResponse, BasicFilterOutputs, BasicFilter, CompareAndDescribeResponse, CompareAndDescribeOutputs, CompareAndDescribe

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

def build_response_compare_and_describe(context):
    outputImage = OutputImage(value=context.image)
    outputText =OutputText(value=context.text)
    compareAndDescribeOutputs = CompareAndDescribeOutputs(outputImage=outputImage,outputText=outputText)
    compareAndDescribeResponse = CompareAndDescribeResponse(outputs=compareAndDescribeOutputs)
    compareAndDescribe= CompareAndDescribe(value=compareAndDescribeResponse)
    executor = ConfigExecutor(value=compareAndDescribe)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel






