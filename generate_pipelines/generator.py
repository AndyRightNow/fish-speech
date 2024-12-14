from utils import get_intermediate_output_base_name, generate_pipelines_logger as logger
from tools.llama.generate_class import LlamaSemanticTokensGenerator
from tools.vqgan.inference_class import VQGanInference
from typing import Any
from os import path
from random import uniform
import constants
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
                logger=logger
            )
        else:
            logger.debug(
                "no_semantic_tokens=True specified. Skipped semantic tokens models loading.")
        if not no_audio:
            self.__audio_generator = VQGanInference()
        else:
            logger.debug(
                "no_audio=True specified. Skipped audio models loading.")

        self.__no_audio = no_audio
        self.__no_semantic_tokens = no_semantic_tokens

    def generate(
        self,
        input_lines,
        input_hash,
        no_audio=None,
        no_semantic_tokens=None,
        force=False,
        segment_title='',
        max_sem_tokens_retries=3
    ):
        output_base_sem_tokens_dir = path.join(
            sem_tokens_output_dir, input_hash
        )
        output_base_audio_dir = path.join(
            constants.audio_output_dir, input_hash)
        Path(output_base_sem_tokens_dir).mkdir(parents=True, exist_ok=True)
        Path(output_base_audio_dir).mkdir(parents=True, exist_ok=True)

        output_file_name = get_intermediate_output_base_name(
            input_lines[0][0], input_lines[-1][0])
        output_base_sem_tokens_name = path.join(
            output_base_sem_tokens_dir, output_file_name
        )
        output_npy_name = f"{output_base_sem_tokens_name}_0.npy"

        no_audio = no_audio if no_audio is not None else self.__no_audio
        no_semantic_tokens = no_semantic_tokens if no_semantic_tokens is not None else self.__no_semantic_tokens

        if not no_semantic_tokens or force:
            if not Path(output_npy_name).is_file() or force:
                logger.debug(
                    f"{output_npy_name} doesn't exist, start generating..")

                genearte_success = False
                generate_count = 0
                generate_exception = None

                while not genearte_success and generate_count < max_sem_tokens_retries:
                    try:
                        self.__sem_tokens_generator.generate(
                            text='\n'.join(([] if not segment_title else [segment_title]) +
                                           ([line for (_, line) in input_lines])),
                            output_name=output_base_sem_tokens_name,
                            temperature=uniform(0.7, 0.9),
                            top_p=uniform(0.7, 0.8),
                            repetition_penalty=1.7
                        )
                        genearte_success = True
                        generate_exception = None
                        logger.success(f"{output_npy_name} was generated.")
                    except Exception as e:
                        logger.exception(
                            f"Failed to generate semantic tokens for {output_npy_name} due to: {e}")
                        generate_exception = e
                        generate_count += 1

                if generate_exception:
                    raise generate_exception
            else:
                logger.debug(
                    f"{output_npy_name} already exists. Skipped npy generation.")
        else:
            logger.debug(
                "no_semantic_tokens=True specified. Skipped semantic tokens generation.")

        output_wav_name = path.join(
            output_base_audio_dir, f"{output_file_name}.wav")

        if not no_audio or force:
            if Path(output_npy_name).is_file() and (not Path(output_wav_name).is_file() or force):
                logger.debug(
                    f"{output_wav_name} doesn't exist, start generating..")
                self.__audio_generator.generate_from_npy(
                    input_path=Path(output_npy_name),
                    output_path=Path(output_wav_name)
                )
                logger.success(f"{output_wav_name} was generated.")
            else:
                logger.debug(
                    f"{output_wav_name} already exists. Skipped audio generation.")
        else:
            logger.debug("no_audio=True specified. Skipped audio generation.")

        if force:
            logger.warning(
                f"Forced the generation of {output_npy_name} and {output_wav_name}")
