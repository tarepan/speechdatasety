"""Adress handling helpers"""


from typing import Optional, Tuple
from pathlib import Path

from speechcorpusy.interface import ItemId


def dataset_adress(
    adress_archive_root: Optional[str],
    corpus_name: str,
    dataset_type: str,
    preprocess_args: str,
    ) -> Tuple[str, Path]:
    """Path of dataset archive file and contents directory.

    Args:
        adress_archive_root:
        corpus_name:
        dataset_type:
        preprocess_args:
    Returns: [archive file adress, contents directory path]
    """
    # Design Notes:
    #   Why not `Path` object? -> Archive adress could be remote url
    #
    # Original Data (corpus) / Prepared Data (dataset) / Transformation (preprocss)
    #   If use different original data, everything change.
    #   Original item can be transformed into different type of data.
    #   Even if data type is same, value could be changed by processing parameters.
    #
    # Directory structure:
    #     datasets/{corpus_name}/{dataset_type}/
    #         archive/{preprocess_args}.zip
    #         contents/{preprocess_args}/{actual_data_here}

    # Contents: Placed under default local directory
    contents_root = local_root = "./tmp"
    # Archive: Placed under given adress or default local directory
    archive_root = adress_archive_root or local_root

    rel_dataset = f"datasets/{corpus_name}/{dataset_type}"
    archive_file = f"{archive_root}/{rel_dataset}/archive/{preprocess_args}.zip"
    contents_dir = f"{contents_root}/{rel_dataset}/contents/{preprocess_args}"
    return archive_file, Path(contents_dir)


def generate_path_getter(data_name: str, dir_dataset: Path):
    """Generate getter of dataset's datum path"""

    def path_getter(item_id: ItemId) -> Path:
        file_name = f"{item_id.name}.{data_name}.pt"
        return dir_dataset / f"{item_id.speaker}" / f"{data_name}s" / file_name

    return path_getter
