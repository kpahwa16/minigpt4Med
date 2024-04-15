import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
# from minigpt4.datasets.datasets.laion_dataset import LaionDataset
# from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
# from minigpt4.datasets.datasets.text_caps import TextCapDataset
# from minigpt4.datasets.datasets.llava_dataset import LlavaDetailDataset, LlavaReasonDataset, LlavaConversationDataset
# from minigpt4.datasets.datasets.unnatural_instruction import UnnaturalDataset
# from minigpt4.datasets.datasets.multitask_conversation import MultiTaskConversationDataset
# from minigpt4.datasets.datasets.flickr import GroundedDetailDataset,CaptionToObjectDataset,PhraseToObjectDataset
# from minigpt4.datasets.datasets.vg_dataset import ReferVisualGenomeDataset
# from minigpt4.datasets.datasets.coco_dataset import ReferCOCODataset, InvReferCOCODataset
# from minigpt4.datasets.datasets.gqa_datasets import GQADataset
# from minigpt4.datasets.datasets.aok_vqa_datasets import AOKVQADataset
from minigpt4.datasets.datasets.pmc_vqa_datasets import PMCVQADataset
# from minigpt4.datasets.datasets.ocrvqa_dataset import OCRVQADataset
from minigpt4.datasets.datasets.pmc_caption import PMCCapDataset
# from minigpt4.datasets.datasets.pmc_captions_dataset import PMCVQADataset


@registry.register_builder("pmc_caption")
class PMCCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = PMCCapDataset

    DATASET_CONFIG_DICT = {"default": "/data/kpahwa/random/MiniGPT-4/minigpt4/configs/datasets/pmc/caption.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
        vis_processor=self.vis_processors["train"],
        text_processor=self.text_processors["train"],
        ann_path=os.path.join(storage_path, 'pmcvqa_captions.json'),  # Correct keyword argument
        vis_root=os.path.join(storage_path, 'figures'),
        )

        return datasets


@registry.register_builder("pmc_vqa")
class PMCVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PMCVQADataset

    DATASET_CONFIG_DICT = {"default": "/data/kpahwa/random/MiniGPT-4/minigpt4/configs/datasets/pmc/defaults_vqa.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'pmc_vqa_dataset.json')],
            vis_root=os.path.join(storage_path, 'figures'),
        )

        return datasets