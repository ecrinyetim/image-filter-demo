
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
    # 1. Önce standart nesneleri oluşturuyoruz
    outputImage_obj = OutputImage(value=context.image)  # context.image LİSTE olmalı
    outputText_obj = OutputText(value=context.text)

    outputs_obj = CompareAndDescribeOutputs(
        outputImage=outputImage_obj,
        outputText=outputText_obj
    )

    response_obj = CompareAndDescribeResponse(outputs=outputs_obj)

    # Task Config (CompareAndDescribe Config Sınıfı)
    task_config = CompareAndDescribe(value=response_obj)

    executor_obj = ConfigExecutor(value=task_config)
    package_configs = PackageConfigs(executor=executor_obj)

    # 2. Modeli Helper ile inşa et (Metadata vs. için gerekli)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    final_package_model = package.build_model(context)

    # --- KRİTİK MÜDAHALE (MANUEL ENJEKSİYON) ---
    # Helper'ın veriyi doğru yazdığından emin değiliz, bu yüzden
    # oluşturduğumuz dolu config yapısını final modele zorla atıyoruz.
    try:
        # Pydantic modellerinde hiyerarşi şöyledir: Model -> Configs -> Executor -> Value (Task) -> Value (Response) -> Outputs
        final_package_model.configs.executor.value.value.outputs.outputImage.value = context.image
        final_package_model.configs.executor.value.value.outputs.outputText.value = context.text
    except Exception as e:
        # Eğer yapı henüz tam oluşmadıysa, komple config'i atayalım
        print(f"Manual injection fallback triggered: {e}")
        final_package_model.configs = package_configs

    return final_package_model




