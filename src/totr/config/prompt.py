from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# From https://github.com/StonyBrookNLP/ircot/blob/3c1820f698eea5eeddb4fba3c56b64c961e063e4/run.py
dataset_prompt_examples: Dict[str, Dict[str, List[str]]] = {
    "hotpotqa": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    },
    "2wikimultihopqa": {
        "1": [
            "228546780bdd11eba7f7acde48001122",
            "97954d9408b011ebbd84ac1f6bf848b6",
            "a5995da508ab11ebbd82ac1f6bf848b6",
            "1ceeab380baf11ebab90acde48001122",
            "35bf3490096d11ebbdafac1f6bf848b6",
            "f86b4a28091711ebbdaeac1f6bf848b6",
            "f44939100bda11eba7f7acde48001122",
            "e5150a5a0bda11eba7f7acde48001122",
            "c6805b2908a911ebbd80ac1f6bf848b6",
            "13cda43c09b311ebbdb0ac1f6bf848b6",
            "f1ccdfee094011ebbdaeac1f6bf848b6",
            "028eaef60bdb11eba7f7acde48001122",
            "8727d1280bdc11eba7f7acde48001122",
            "79a863dc0bdc11eba7f7acde48001122",
            "c6f63bfb089e11ebbd78ac1f6bf848b6",
        ],
        "2": [
            "c6805b2908a911ebbd80ac1f6bf848b6",
            "5897ec7a086c11ebbd61ac1f6bf848b6",
            "028eaef60bdb11eba7f7acde48001122",
            "af8c6722088b11ebbd6fac1f6bf848b6",
            "1ceeab380baf11ebab90acde48001122",
            "5811079c0bdc11eba7f7acde48001122",
            "228546780bdd11eba7f7acde48001122",
            "e5150a5a0bda11eba7f7acde48001122",
            "f44939100bda11eba7f7acde48001122",
            "f1ccdfee094011ebbdaeac1f6bf848b6",
            "13cda43c09b311ebbdb0ac1f6bf848b6",
            "79a863dc0bdc11eba7f7acde48001122",
            "a5995da508ab11ebbd82ac1f6bf848b6",
            "cdbb82ec0baf11ebab90acde48001122",
            "c6f63bfb089e11ebbd78ac1f6bf848b6",
        ],
        "3": [
            "028eaef60bdb11eba7f7acde48001122",
            "8727d1280bdc11eba7f7acde48001122",
            "79a863dc0bdc11eba7f7acde48001122",
            "4724c54e08e011ebbda1ac1f6bf848b6",
            "e5150a5a0bda11eba7f7acde48001122",
            "35bf3490096d11ebbdafac1f6bf848b6",
            "a5995da508ab11ebbd82ac1f6bf848b6",
            "228546780bdd11eba7f7acde48001122",
            "97954d9408b011ebbd84ac1f6bf848b6",
            "f44939100bda11eba7f7acde48001122",
            "1ceeab380baf11ebab90acde48001122",
            "f86b4a28091711ebbdaeac1f6bf848b6",
            "c6f63bfb089e11ebbd78ac1f6bf848b6",
            "af8c6722088b11ebbd6fac1f6bf848b6",
            "5897ec7a086c11ebbd61ac1f6bf848b6",
        ],
    },
    "musique": {
        "1": [
            "2hop__804754_52230",
            "2hop__292995_8796",
            "2hop__496817_701819",
            "2hop__154225_727337",
            "2hop__642271_608104",
            "2hop__439265_539716",
            "2hop__195347_20661",
            "2hop__131516_53573",
            "2hop__427213_79175",
            "3hop1__443556_763924_573834",
            "2hop__782642_52667",
            "2hop__861128_15822",
            "4hop3__703974_789671_24078_24137",
            "3hop1__61746_67065_43617",
            "4hop3__463724_100414_35260_54090",
        ],
        "2": [
            "2hop__292995_8796",
            "2hop__154225_727337",
            "2hop__642271_608104",
            "2hop__195347_20661",
            "3hop1__61746_67065_43617",
            "2hop__861128_15822",
            "3hop1__753524_742157_573834",
            "2hop__496817_701819",
            "4hop3__703974_789671_24078_24137",
            "3hop1__858730_386977_851569",
            "2hop__804754_52230",
            "2hop__782642_52667",
            "2hop__102217_58400",
            "2hop__387702_20661",
            "3hop1__443556_763924_573834",
        ],
        "3": [
            "2hop__427213_79175",
            "3hop1__753524_742157_573834",
            "2hop__782642_52667",
            "2hop__496817_701819",
            "3hop1__443556_763924_573834",
            "4hop3__463724_100414_35260_54090",
            "2hop__292995_8796",
            "2hop__804754_52230",
            "3hop1__858730_386977_851569",
            "2hop__131516_53573",
            "2hop__387702_20661",
            "4hop3__703974_789671_24078_24137",
            "2hop__154225_727337",
            "3hop1__61746_67065_43617",
            "2hop__642271_608104",
        ],
    },
    "iirc": {
        "1": [
            "q_10344",
            "q_10227",
            "q_9591",
            "q_3283",
            "q_8776",
            "q_8981",
            "q_9518",
            "q_1672",
            "q_9499",
            "q_8173",
            "q_9433",
            "q_8350",
            "q_3268",
            "q_8736",
            "q_389",
        ],
        "2": [
            "q_9499",
            "q_10236",
            "q_2466",
            "q_10270",
            "q_8776",
            "q_9591",
            "q_10227",
            "q_8981",
            "q_9518",
            "q_3290",
            "q_8173",
            "q_8736",
            "q_10344",
            "q_389",
            "q_1672",
        ],
        "3": [
            "q_10344",
            "q_10227",
            "q_8776",
            "q_3268",
            "q_3283",
            "q_10270",
            "q_10236",
            "q_8736",
            "q_1672",
            "q_3208",
            "q_9433",
            "q_8350",
            "q_9591",
            "q_8981",
            "q_3290",
        ],
    },
}


@dataclass
class PromptConfig:
    prompt_directory: str
    prompt_dataset: str
    react_prompt_filename: str
    cot_prompt_filename: str
    direct_prompt_filename: str
    prompt_set: str

    @property
    def prompt_example_ids(self) -> List[str]:
        return dataset_prompt_examples[self.prompt_dataset][self.prompt_set]

    @property
    def cot_prompt_file(self) -> Path:
        return Path(
            self.prompt_directory, self.prompt_dataset, self.cot_prompt_filename
        ).resolve()

    @property
    def react_prompt_file(self) -> Path:
        return Path(
            self.prompt_directory, self.prompt_dataset, self.react_prompt_filename
        ).resolve()

    @property
    def direct_prompt_file(self) -> Path:
        return Path(
            self.prompt_directory, self.prompt_dataset, self.direct_prompt_filename
        ).resolve()
