from tools.llama.generate_class import LlamaSemanticTokensGenerator
from tools.vqgan.inference_class import VQGanInference
from typing import Any
from os import path
from random import uniform
import constants
from loguru import logger
from pathlib import Path

sem_tokens_output_dir = path.join(constants.base_output_dir, "semantic_tokens")


class TTSGenerator:
    __sem_tokens_generator: Any = None
    __audio_generator: Any = None
    __no_audio = False
    __no_semantic_tokens = False

    def __init__(self, prompt_text, prompt_tokens, no_audio=False, no_semantic_tokens=False):
        if not no_semantic_tokens:
            self.__sem_tokens_generator = LlamaSemanticTokensGenerator(
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
            )
        else:
            logger.info("no_semantic_tokens=True specified. Skipped semantic tokens models loading.")
        if not no_audio:
            self.__audio_generator = VQGanInference()
        else:
            logger.info("no_audio=True specified. Skipped audio models loading.")

        self.__no_audio = no_audio
        self.__no_semantic_tokens = no_semantic_tokens

    def generate(self, input_lines, input_hash):
        output_file_name = f"{input_hash}_{input_lines[0][0]}_{input_lines[-1][0]}"
        output_base_sem_tokens_name = path.join(
            sem_tokens_output_dir, output_file_name
        )
        output_npy_name = f"{output_base_sem_tokens_name}_0.npy"

        if not self.__no_semantic_tokens:
            if not Path(output_npy_name).is_file():
                logger.info(
                    f"{output_npy_name} doesn't exist, start generating..")
                self.__sem_tokens_generator.generate(
                    text='\n'.join(
                        [line for (_, line) in input_lines]),
                    output_name=output_base_sem_tokens_name,
                    temperature=uniform(0.7, 1.5),
                    repetition_penalty=uniform(1.2, 1.6),
                    top_p=uniform(0.7, 0.8),
                )
            else:
                logger.success(
                    f"{output_npy_name} already exists. Skipped npy generation.")

        output_wav_name = path.join(
            constants.audio_output_dir, f"{output_file_name}.wav")

        if not self.__no_audio:
            if Path(output_npy_name).is_file() and not Path(output_wav_name).is_file():
                logger.info(
                    f"{output_wav_name} doesn't exist, start generating..")
                self.__audio_generator.generate_from_npy(
                    input_path=Path(output_npy_name),
                    output_path=Path(path.join(
                        constants.audio_output_dir, f"{output_file_name}.wav"))
                )
            else:
                logger.success(
                    f"{output_wav_name} already exists. Skipped audio generation.")
