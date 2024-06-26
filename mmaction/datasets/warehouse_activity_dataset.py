"""COPYRIGHT (C) [2024], CompScience, Inc.

This software is proprietary and confidential. Unauthorized copying,
distribution, modification, or use is strictly prohibited. This software
is provided "as is," without warranty of any kind.
"""
import os
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from mmengine.fileio import exists, list_from_file, load

from mmaction.evaluation import read_labelmap
from mmaction.registry import DATASETS
from mmaction.samplers import WeightedSampler
from mmaction.utils import ConfigType
from .ava_dataset import AVADataset


@DATASETS.register_module()
class WarehouseActivityDataset(AVADataset):
    """STAD dataset for spatial temporal action detection.

    The dataset loads raw frames/video files, bounding boxes,
    proposals and applies specified transformations to return
    a dict containing the frame tensors and other information.

    This datasets can load information from the following files:

    .. code-block:: txt

        ann_file -> ava_{train, val}_{v2.1, v2.2}.csv
        exclude_file -> ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv
        label_file -> ava_action_list_{v2.1, v2.2}.pbtxt /
                      ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt
        proposal_file -> ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl

    Particularly, the proposal_file is a pickle file which contains
    ``img_key`` (in format of ``{video_id},{timestamp}``). Example of a pickle
    file:

    .. code-block:: JSON

        {
            ...
            '0f39OWEqJ24,0902':
                array([[0.011   , 0.157   , 0.655   , 0.983   , 0.998163]]),
            '0f39OWEqJ24,0912':
                array([[0.054   , 0.088   , 0.91    , 0.998   , 0.068273],
                       [0.016   , 0.161   , 0.519   , 0.974   , 0.984025],
                       [0.493   , 0.283   , 0.981   , 0.984   , 0.983621]]),
            ...
        }

    Args:
        ann_file (str): Path to the annotation file like
            ``ava_{train, val}_{v2.1, v2.2}.csv``.
        exclude_file (str): Path to the excluded timestamp file like
            ``ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv``.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        label_file (str): Path to the label file like
            ``ava_action_list_{v2.1, v2.2}.pbtxt`` or
            ``ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt``.
            Defaults to None.
        filename_tmpl (str): Template for each filename.
            Defaults to 'img_{:05}.jpg'.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. It should be set to 1 for AVA, since
            frame index start from 1 in AVA dataset. Defaults to 1.
        proposal_file (str): Path to the proposal file like
            ``ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl``.
            Defaults to None.
        person_det_score_thr (float): The threshold of person detection scores,
            bboxes with scores above the threshold will be used.
            Note that 0 <= person_det_score_thr <= 1. If no proposal has
            detection score larger than the threshold, the one with the largest
            detection score will be used. Default: 0.9.
        num_classes (int): The number of classes of the dataset. Default: 81.
            (AVA has 80 action classes, another 1-dim is added for potential
            usage)
        custom_classes (List[int], optional): A subset of class ids from origin
            dataset. Please note that 0 should NOT be selected, and
            ``num_classes`` should be equal to ``len(custom_classes) + 1``.
        data_prefix (dict or ConfigDict): Path to a directory where video
            frames are held. Defaults to ``dict(img='')``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        modality (str): Modality of data. Support ``RGB``, ``Flow``.
            Defaults to ``RGB``.
        num_max_proposals (int): Max proposals number to store.
            Defaults to 1000.
        timestamp_start (int): The start point of included timestamps. The
            default value is referred from the official website.
            Defaults to 902.
        timestamp_end (int): The end point of included timestamps. The default
            value is referred from the official website. Defaults to 1798.
        use_frames (bool): Whether to use rawframes as input.
            Defaults to True.
        fps (int): Overrides the default FPS for the dataset. If set to 1,
            means counting timestamp by frame, e.g. MultiSports dataset.
            Otherwise by second. Defaults to 30.
        multilabel (bool): Determines whether it is a multilabel recognition
            task. Defaults to True.
    """
    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 exclude_file: Optional[str] = None,
                 label_file: Optional[str] = None,
                 filename_tmpl: str = 'img_{:05}.jpg',
                 start_index: int = 1,
                 proposal_file: str = None,
                 person_det_score_thr: float = 0.9,
                 num_classes: int = 81,
                 custom_classes: Optional[List[int]] = None,
                 data_prefix: ConfigType = dict(img=''),
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 num_max_proposals: int = 1000,
                 timestamp_start: int = 900,
                 timestamp_end: int = 1800,
                 use_frames: bool = True,
                 fps: int = 30,
                 fps_file: Optional[str] = None,
                 multilabel: bool = True,
                 class_weights: Optional[dict] = None,
                 augment_labels: Optional[bool] = False,
                 **kwargs) -> None:
        if fps_file is not None:
            fps_mapping = pd.read_csv(fps_file)
            self._FPS = self.create_fps_mapping(fps_mapping)
            self._FPS['default'] = fps
        else:
            self._FPS = fps  # Keep this as standard

        self.augment_labels = augment_labels
        self.class_weights = class_weights
        self.per_sample_weights = None
        if class_weights:
            self.per_sample_weights = []

        self.custom_classes = custom_classes
        if custom_classes is not None:
            assert num_classes == len(custom_classes) + 1
            assert 0 not in custom_classes
            _, class_whitelist = read_labelmap(open(label_file))
            assert set(custom_classes).issubset(class_whitelist)

            self.custom_classes = list([0] + custom_classes)
        self.exclude_file = exclude_file
        self.label_file = label_file
        self.proposal_file = proposal_file
        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0, 1]. ')
        self.person_det_score_thr = person_det_score_thr
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.num_max_proposals = num_max_proposals
        self.filename_tmpl = filename_tmpl
        self.use_frames = use_frames
        self.multilabel = multilabel

        super(AVADataset, self).__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs)

        if self.proposal_file is not None:
            self.proposals = load(self.proposal_file)
        else:
            self.proposals = None

    def create_fps_mapping(self, fps_mapping_df: pd.DataFrame) -> dict:
        fps_mapping = {}
        for _, row in fps_mapping_df.iterrows():
            fps_mapping[row['image_name']] = int(round(row['fps']))
        return fps_mapping

    def _get_frame_id_from_filename(self, filename):
        """Extract frame id from filename."""
        return int(osp.splitext(osp.basename(filename))[0].split('_')[-1])

    def _get_num_frames(self, video_id):
        img_root = self.data_prefix['img']
        target_dir = osp.join(img_root, video_id)
        return len(os.listdir(target_dir))

    def _convert_one_hot_to_label(self, one_hot_labels):
        """Convert one hot labels to label."""
        return np.where(one_hot_labels == 1)[1]

    def _duplicate_labels_within_frame(
        self,
        bboxes: np.ndarray,
        labels: np.ndarray,
        entity_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each label obj in labels is a list of actions performed
        by the same person in the same frame.
        Check if the labels needs to be augmented
        by finding if there is a rare classes in the label
        If there are multiple rare classes, we need to
        find the maximum class weight aka. number of duplications

        Returns:
            np.ndarray: Augmented bboxes
            np.ndarray: Augmented labels
            np.ndarray: Augmented entity_ids
        """
        # Augment rare labels by duplicating samples with the same label
        aug_bboxes, aug_labels, aug_entity_ids = [], [], []
        for i, label in enumerate(labels):
            # Add original samples
            aug_bboxes.append(bboxes[i])
            aug_labels.append(labels[i])
            aug_entity_ids.append(entity_ids[i])

            # label is a one-hot vector [0, 0, 1, 1, 0, ...]
            label_idxs = np.where(label == 1)[0]
            weights = [self.class_weights[self.custom_classes[int(idx)]] for idx in label_idxs]

            # Each label obj is a list of actions performed
            # by the same person in the same frame.
            # Check if the labels needs to be augmented
            # by finding if there is a rare classes in the label
            # If there are multiple rare classes, we need to
            # find the maximum class weight aka. number of duplications
            duplication_factor = int(max(max(weights), 1))

            # If the class weight is less than 2, we don't need to augment
            for _ in range(duplication_factor - 1):
                aug_bboxes.append(bboxes[i])
                aug_labels.append(labels[i])
                aug_entity_ids.append(entity_ids[i])
        return np.array(aug_bboxes), np.array(aug_labels), np.array(aug_entity_ids)

    def build_shot_info_dict(self) -> dict:
        """Compute shot info for a video."""
        exists(self.ann_file)
        df = pd.read_csv(self.ann_file)
        try:
            df.columns = [
                "video_name",
                "middle_frame_timestamp",
                "x1",
                "y1",
                "x2",
                "y2",
                "class_label",
                "person_id",
                "obj_hash",
                "created_by",
                "created_at",
            ]
        except:
            df.columns = [
                "video_name",
                "middle_frame_timestamp",
                "x1",
                "y1",
                "x2",
                "y2",
                "class_label",
                "person_id",
            ]
        img_prefix = self.data_prefix['img']
        shot_info_dict = {}
        unique_video_names = df['video_name'].unique()
        for video_name in unique_video_names:
            file_list = sorted(os.listdir(osp.join(img_prefix, video_name)))
            first_file = file_list[0]
            last_file = file_list[-1]
            start = self._get_frame_id_from_filename(first_file)
            end = self._get_frame_id_from_filename(last_file)
            shot_info_dict[video_name] = (start, end)
        return shot_info_dict

    def load_data_list(self) -> List[dict]:
        """Load AVA annotations."""
        exists(self.ann_file)
        data_list = []
        records_dict_by_img = defaultdict(list)
        fin = list_from_file(self.ann_file)
        self.shot_info_dict = self.build_shot_info_dict()
        for line in fin:
            line_split = line.strip().split(',')

            label = int(line_split[6])
            if self.custom_classes is not None:
                if label not in self.custom_classes:
                    continue
                label = self.custom_classes.index(label)

            video_id = line_split[0]
            timestamp = int(line_split[1])  # count by second or frame.
            img_key = f'{video_id},{timestamp:05d}'

            entity_box = np.array(list(map(float, line_split[2:6])))
            entity_id = int(line_split[7])
            if self.use_frames:
                # # Load shot info based on num files of target folder
                shot_info = self.shot_info_dict[video_id]
            # for video data, automatically get shot info when decoding
            else:
                shot_info = None

            self.shot_info_dict[video_id] = shot_info

            video_info = dict(
                video_id=video_id,
                timestamp=timestamp,
                entity_box=entity_box,
                label=label,
                entity_id=entity_id,
                shot_info=shot_info)
            records_dict_by_img[img_key].append(video_info)

        for img_key in records_dict_by_img:
            video_id, timestamp = img_key.split(',')
            bboxes, labels, entity_ids = self.parse_img_record(
                records_dict_by_img[img_key])

            if self.augment_labels and self.class_weights:
                (
                    aug_bboxes,
                    aug_labels,
                    aug_entity_ids,
                ) = self._duplicate_labels_within_frame(bboxes, labels, entity_ids)
                bboxes = aug_bboxes
                labels = aug_labels
                entity_ids = aug_entity_ids

            ann = dict(
                gt_bboxes=bboxes, gt_labels=labels, entity_ids=entity_ids)
            frame_dir = video_id
            if self.data_prefix['img'] is not None:
                frame_dir = osp.join(self.data_prefix['img'], frame_dir)

            shot_info = self.shot_info_dict[video_id]
            video_info = dict(
                frame_dir=frame_dir,
                video_id=video_id,
                timestamp=int(timestamp),
                img_key=img_key,
                shot_info=shot_info,
                fps=self._FPS.get(video_id, self._FPS['default']),
                ann=ann)
            if not self.use_frames:
                video_info['filename'] = video_info.pop('frame_dir')

            # each record dict contains labels for all objects in frames
            # which means it includes multiple labels
            # The weights are calculated as the sum of the weights of all labels
            converted_labels = self._convert_one_hot_to_label(labels)
            original_labels = [self.custom_classes[label] for label in converted_labels]
            # with open("sampling-weights-weighted.txt", "a") as f:
            #     f.write(",".join([str(label) for label in original_labels]))
            #     f.write("\n")

            if self.class_weights:
                # Also, the labels are the index of the class in the custom_classes list
                # so we need to map them to the original labels
                converted_weights = [self.class_weights[self.custom_classes[label]] for label in converted_labels]
                total_weight = sum(converted_weights)
                self.per_sample_weights.append(total_weight)
            else:
                converted_weights = [1.0] * len(converted_labels)
                total_weight = len(converted_labels)
            data_list.append(video_info)

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super(AVADataset, self).get_data_info(idx)
        img_key = data_info['img_key']

        data_info['filename_tmpl'] = self.filename_tmpl
        data_info['timestamp_start'] = self.timestamp_start
        data_info['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                raise ValueError(f'{img_key} not in proposals.')
                data_info['proposals'] = np.array([[0, 0, 1, 1]])
                data_info['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    data_info['proposals'] = proposals[:, :4]
                    data_info['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    data_info['proposals'] = proposals

                assert data_info['proposals'].max() <= 1 and \
                    data_info['proposals'].min() >= 0, \
                    (f'relative proposals invalid: max value '
                     f'{data_info["proposals"].max()}, min value '
                     f'{data_info["proposals"].min()}')

        ann = data_info.pop('ann')
        data_info['gt_bboxes'] = ann['gt_bboxes']
        data_info['gt_labels'] = ann['gt_labels']
        data_info['entity_ids'] = ann['entity_ids']

        return data_info
