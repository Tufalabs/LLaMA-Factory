from enum import Enum


class DatasetName(Enum):
    OPEN_THOUGHTS_2 = "open-thoughts/OpenThoughts2-1M"
    WEB_BACK_TRANSLATION = "YuxinJiang/web_back_translation_100k_llama3"
    WEBR_BASIC = "YuxinJiang/WebR-Basic-100k"
    WEBR_PRO = "YuxinJiang/WebR-Pro-100k"

    @property
    def full_path(self) -> str:
        return self.value

    @property
    def dataset_name(self) -> str:
        _source, name = self.value.split("/")
        return name
