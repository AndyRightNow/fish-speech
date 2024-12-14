
import click
from loguru import _logger
import constants
import sys
import os


def use_shared_command_options(func):
    func = click.option(
        "--prompt-name",
        type=str,
        required=True
    )(func)
    func = click.option(
        "--input-name",
        type=str,
        required=True
    )(func)
    func = click.option(
        "--force-segment-index",
        type=int,
        multiple=True,
        default=[]
    )(func)
    func = click.option(
        "--max-sem-input-count",
        type=int,
        default=constants.default_max_semantic_tokens_input_count
    )(func)
    func = click.option(
        "--start-segment-index",
        type=int,
        default=0
    )(func)

    return func


def get_intermediate_output_base_name(start_segment_index, end_segment_index):
    return f"{start_segment_index}_{end_segment_index}"


generate_pipelines_logger = _logger.Logger(
    core=_logger.Core(),
    exception=None,
    depth=0,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patchers=[],
    extra={},
)

generate_pipelines_logger.configure(
    handlers=[
        dict(
            sink=sys.stderr,
            format="<cyan>[GeneratePipelines]</cyan> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        dict(
            sink='generate_pipelines/logs/generate_pipelines_log_{time}.log',
            format="[GeneratePipelines] | {level: <8} | {name}:{function}:{line} - {message} - {time:YYYY-MM-DD HH:mm:ss.SSS}"
        )
    ]
)
