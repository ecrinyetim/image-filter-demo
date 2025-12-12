
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
    # 1. Output Nesneleri
    # context.image artık bir LİSTE olduğu için OutputImage bunu kabul edecektir.
    outputImage_obj = OutputImage(value=context.image)
    outputText_obj = OutputText(value=context.text)

    # 2. Outputs Konteyneri
    # Değişken adını sınıf adından ayırdık (camelCase riskine girmeyelim)
    outputs_obj = CompareAndDescribeOutputs(
        outputImage=outputImage_obj,
        outputText=outputText_obj
    )

    # 3. Response Nesnesi
    response_obj = CompareAndDescribeResponse(outputs=outputs_obj)

    # 4. Config/Task Nesnesi
    # BURASI KRİTİK: 'compareAndDescribe' değişken ismini 'task_config' yaptık.
    # Böylece imported edilen 'CompareAndDescribe' sınıfı ile çakışmaz.
    task_config = CompareAndDescribe(value=response_obj)

    # 5. Executor
    executor_obj = ConfigExecutor(value=task_config)

    # 6. Package Configs
    package_configs = PackageConfigs(executor=executor_obj)

    # 7. Modeli İnşa Et
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    packageModel = package.build_model(context)

    return packageModel





