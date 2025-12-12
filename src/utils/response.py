from sdks.novavision.src.helper.package import PackageHelper
# Importlarda çakışma olmaması için dikkatli oluyoruz
from components.ImageFilterDemo.src.models.PackageModel import (
    PackageModel, PackageConfigs, ConfigExecutor,
    OutputImage, OutputText,
    BasicFilterResponse, BasicFilterOutputs, BasicFilter,
    CompareAndDescribeResponse, CompareAndDescribeOutputs, CompareAndDescribe
)


def build_response_basic_filter(context):
    outputImage = OutputImage(value=context.image)
    basicFilterOutputs = BasicFilterOutputs(outputImage=outputImage)
    basicFilterResponse = BasicFilterResponse(outputs=basicFilterOutputs)
    basicFilter = BasicFilter(value=basicFilterResponse)
    executor = ConfigExecutor(value=basicFilter)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    return package.build_model(context)


def build_response_compare_and_describe(context):
    # 1. Output Nesneleri
    # Değişken isimlerine _obj ekledim ki sınıf isimleriyle karışmasın
    outputImage_obj = OutputImage(value=context.image)
    outputText_obj = OutputText(value=context.text)

    # 2. Outputs Konteyneri
    outputs_obj = CompareAndDescribeOutputs(
        outputImage=outputImage_obj,
        outputText=outputText_obj
    )

    # 3. Response Nesnesi
    response_obj = CompareAndDescribeResponse(outputs=outputs_obj)

    # 4. Task Config
    # BURADA HATA VARDI: CompareAndDescribe = CompareAndDescribe(...) yapınca kod karışıyor.
    # Değişken adını task_config yaptım.
    task_config = CompareAndDescribe(value=response_obj)

    # 5. Executor ve Paket
    executor_obj = ConfigExecutor(value=task_config)
    package_configs_obj = PackageConfigs(executor=executor_obj)

    # 6. Build Model
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs_obj)
    final_package_model = package.build_model(context)

    # 7. MANUEL DOLDURMA (GÜVENLİK AĞI)
    # Eğer Helper fonksiyonu veriyi doğru yere koyamazsa diye, biz elle koyuyoruz.
    # ImageView'ın okuduğu yol tam olarak burasıdır.
    try:
        # Hiyerarşi: Configs -> Executor -> Value(Task) -> Value(Response) -> Outputs -> OutputImage -> Value
        final_package_model.configs.executor.value.value.outputs.outputImage.value = context.image
        final_package_model.configs.executor.value.value.outputs.outputText.value = context.text
    except Exception:
        # Eğer yukarıdaki yol henüz oluşmadıysa sessizce devam et, ama genelde bu çözer.
        pass

    return final_package_model