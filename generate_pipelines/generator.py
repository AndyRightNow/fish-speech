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

    def __init__(self, prompt_text, prompt_tokens):
        self.__sem_tokens_generator = LlamaSemanticTokensGenerator(
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            logger=logger
        )

        self.__audio_generator = VQGanInference()

    def generate(
        self,
        input_lines,
        input_hash,
        no_audio=None,
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

        no_audio = no_audio if no_audio is not None else self.__no_audio
        output_wav_name = path.join(
            output_base_audio_dir, f"{output_file_name}.wav")

        if not no_audio or force:
            genearte_success = False
            generate_count = 0
            generate_exception = None

            while not genearte_success and generate_count < max_sem_tokens_retries:
                try:
                    sem_token_generator = self.__sem_tokens_generator.generate(
                        text='\n'.join(([] if not segment_title else [segment_title]) +
                                       ([line for (_, line) in input_lines])),
                        temperature=uniform(0.7, 0.9),
                        top_p=uniform(0.7, 0.8),
                        repetition_penalty=1.7
                    )
                    generated_sem_tokens = next(sem_token_generator)
                    genearte_success = True
                    generate_exception = None

                    self.__audio_generator.generate_from_npy(
                        input_indices=generated_sem_tokens,
                        output_path=Path(output_wav_name)
                    )

                    logger.success(f"{output_wav_name} was generated.")
                    return output_wav_name
                except Exception as e:
                    logger.exception(
                        f"Failed to generate audio for {output_wav_name} due to: {e}")
                    generate_exception = e
                    generate_count += 1

            if generate_exception:
                raise generate_exception
        else:
            logger.debug("no_audio=True specified. Skipped audio generation.")

        if force:
            logger.warning(
                f"Forced the generation of {output_wav_name}")

        return None
