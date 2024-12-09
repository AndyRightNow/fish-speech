
import click
import constants

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