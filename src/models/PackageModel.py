
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config

class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"

class InputImage2(Input):
    name: Literal["inputImage2"] = "inputImage2"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image],Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"



#BasicFilter Configs
class Blur(Config):
    name: Literal["Blur"] = "Blur"
    value: Literal["Blur"] = "Blur"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Blur"

class Sharpen(Config):
    name: Literal["Sharpen"] = "Sharpen"
    value: Literal["Sharpen"] = "Sharpen"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Sharpen"

class FilterType(Config):
    """
    Select which filter to apply to the image.
    """
    name: Literal["filterType"] = "filterType"
    value: Union[Blur, Sharpen]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    class Config:
        title="Filter Type"


class Intensity(Config):
    name: Literal["intensity"] = "intensity"
    value: float = Field(ge=1, le=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

#Threshold Configs
class ThresholdValue(Config):
    name: Literal["thresholdValue"] = "thresholdValue"
    value: int = Field(default=127, ge=0, le=255)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"


#Inputs
class  ThresholdInputs(Inputs):
    inputImage: InputImage
    inputImage2: InputImage2

class BasicFilterInputs(Inputs):
    inputImage: InputImage


#Configs
class ThresholdConfigs(Configs):
    thresholdValue:ThresholdValue

class BasicFilterConfigs(Configs):
    filterType: FilterType
    intensity : Intensity


#Outputs
class OutputText(Output):
    name: Literal["outputText"] = "outputText"
    value:str
    type: Literal["string"] = "string"

class ThresholdOutputs(Outputs):
    outputImage: OutputImage
    outputText: OutputText

class BasicFilterOutputs(Outputs):
    outputImage: OutputImage


#Requests
class ThresholdRequest(Request):
    inputs: Optional[ThresholdInputs]
    configs: ThresholdConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class BasicFilterRequest(Request):
    inputs: Optional[BasicFilterInputs]
    configs: BasicFilterConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


#Responses
class ThresholdResponse(Response):
    outputs: ThresholdOutputs


class BasicFilterResponse(Response):
    outputs: BasicFilterOutputs



#Executors
class Threshold(Config):
    name: Literal["Threshold"] = "Threshold"
    value: Union[ThresholdRequest, ThresholdResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Threshold"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

class BasicFilter(Config):
    name: Literal["BasicFilter"] = "BasicFilter"
    value: Union[BasicFilterRequest, BasicFilterResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "BasicFilter"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

#ImageFilterDemo
class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[BasicFilter,Threshold]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"

class PackageConfigs(Configs):
    executor: ConfigExecutor

class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["ImageFilterDemo"] = "ImageFilterDemo"
