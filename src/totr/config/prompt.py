from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptConfig:
    prompt_directory: str
    prompt_dataset: str
    cot_prompt_filename: str
    direct_prompt_filename: str
    no_context_cot_prompt_filename: str
    no_context_direct_prompt_filename: str

    @property
    def cot_prompt_file(self) -> Path:
        return Path(
            self.prompt_directory, self.prompt_dataset, self.cot_prompt_filename
        ).resolve()

    @property
    def direct_prompt_file(self) -> Path:
        return Path(
            self.prompt_directory, self.prompt_dataset, self.direct_prompt_filename
        ).resolve()

    @property
    def no_context_cot_prompt_file(self) -> Path:
        return Path(
            self.prompt_directory,
            self.prompt_dataset,
            self.no_context_cot_prompt_filename,
        ).resolve()

    @property
    def no_context_direct_prompt_file(self) -> Path:
        return Path(
            self.prompt_directory,
            self.prompt_dataset,
            self.no_context_direct_prompt_filename,
        ).resolve()
