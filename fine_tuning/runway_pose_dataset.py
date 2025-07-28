import os
import json
import cv2
import numpy as np

from typing import List, Tuple, Union

from sklearn.model_selection import train_test_split

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.transforms.keypoint_transforms import AbstractKeypointTransform
from super_gradients.training.samples import PoseEstimationSample
from super_gradients.training.datasets.pose_estimation_datasets.abstract_pose_estimation_dataset import AbstractPoseEstimationDataset

from super_gradients.training.utils.distributed_training_utils import wait_for_the_master
from super_gradients.common.environment.ddp_utils import get_local_rank
from super_gradients.training.datasets.pose_estimation_datasets import YoloNASPoseCollateFN


class RunwayPoseEstimationDataset(AbstractPoseEstimationDataset):
    @classmethod
    def split_runway_pose_dataset(cls, annotation_file, train_annotation_file, val_annotation_file, test_annotation_file, val_fraction,test_fraction):
        """
        Splits the runway pose dataset into training and validation sets.
        :param annotation_file: Path to the original annotation file.
        :param train_annotation_file: Path to save the training annotations.
        :param val_annotation_file: Path to save the validation annotations.
        :param val_fraction: Fraction of the dataset to be used for validation.
        """
        with open(annotation_file, "r") as f:
            annotation = json.load(f)
        

        image_ids = [img["id"] for img in annotation["images"]]
        labels = [[ann["category_id"] for ann in annotation["annotations"] if ann["image_id"] == img_id] for img_id in image_ids]
        labels = [label[0] if len(label) else -1 for label in labels]

        train_ids = []
        val_ids = []
        test_ids = []
        
        for img in annotation["images"]:
            fname = img["file_name"].lower()
            if "train" in fname:
                train_ids.append(img["id"])
            elif "val" in fname:
                val_ids.append(img["id"])
            elif "test" in fname:
                test_ids.append(img["id"])
            else:
                # Optional: handle images without split keyword, e.g., assign to train or ignore
                pass


        #train_ids, temp_val_ids = train_test_split(image_ids, test_size=val_fraction, random_state=42, stratify=labels)
        #val_ids, test_ids = train_test_split(temp_val_ids, test_size=test_fraction, random_state=42, stratify=[labels[image_ids.index(id)] for id in temp_val_ids])

        train_annotations = {
            "info": annotation.get("info", {}),
            "categories": annotation["categories"],
            "images": [img for img in annotation["images"] if img["id"] in train_ids],
            "annotations": [ann for ann in annotation["annotations"] if ann["image_id"] in train_ids],
        }

        val_annotations = {
            "info": annotation.get("info", {}),
            "categories": annotation["categories"],
            "images": [img for img in annotation["images"] if img["id"] in val_ids],
            "annotations": [ann for ann in annotation["annotations"] if ann["image_id"] in val_ids],
        }

        test_annotations = {
            "info": annotation.get("info", {}),
            "categories": annotation["categories"],
            "images": [img for img in annotation["images"] if img["id"] in test_ids],
            "annotations": [ann for ann in annotation["annotations"] if ann["image_id"] in test_ids],
        }

        with open(train_annotation_file, "w") as f:
            json.dump(train_annotations, f, indent=2)
        with open(val_annotation_file, "w") as f:
            json.dump(val_annotations, f, indent=2)
        with open(test_annotation_file, "w") as f:
            json.dump(test_annotations, f, indent=2)

    @resolve_param("transforms", TransformsFactory())
    def __init__(
        self,
        data_dir: str,
        images_dir: str,
        json_file: str,
        transforms: List[AbstractKeypointTransform],
        edge_links: Union[List[Tuple[int, int]], np.ndarray],
        edge_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
        keypoint_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
    ):
        """
        :param data_dir: Root directory of the COCO dataset
        :param images_dir: path suffix to the images directory inside the data_dir
        :param json_file: path suffix to the json file inside the data_dir
        :param include_empty_samples: Not used, but exists for compatibility with COCO dataset config.
        :param target_generator: Target generator that will be used to generate the targets for the model.
            See DEKRTargetsGenerator for an example.
        :param transforms: Transforms to be applied to the image & keypoints
        """
        json_path = os.path.join(data_dir, json_file)
        with open(json_path, "r") as f:
            annotation = json.load(f)

        joints = annotation["categories"][0]["keypoints"]
        num_joints = len(joints)

        super().__init__( 
            transforms=transforms,
            num_joints=num_joints,
            edge_links=edge_links,
            edge_colors=edge_colors,
            keypoint_colors=keypoint_colors,
        )

        self.image_id_to_file = {img["id"]: os.path.join(data_dir, images_dir, img["file_name"]) for img in annotation["images"]}
        self.image_ids = list(self.image_id_to_file.keys())
        self.image_files = list(self.image_id_to_file.values())

        self.annotations = []
        for image_id in self.image_ids:
            anns = [ann for ann in annotation["annotations"] if ann["image_id"] == image_id]
            keypoints_list = []
            bboxes_list = []
            for ann in anns:
                kpts = np.array(ann["keypoints"]).reshape(num_joints, 3)
                x, y, w, h = ann["bbox"]
                keypoints_list.append(kpts)
                bboxes_list.append(np.array([x, y, w, h]))
            if keypoints_list:
                self.annotations.append((np.array(keypoints_list, dtype=np.float32), np.array(bboxes_list, dtype=np.float32)))
            else:
                self.annotations.append((np.zeros((0, num_joints, 3)), np.zeros((0, 4))))

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        :return: Number of samples in the dataset.
        """
        return len(self.image_ids)

    def load_sample(self, index) -> PoseEstimationSample:
        """
        Loads a sample from the dataset.
        :param index: Index of the sample to load.
        :return: PoseEstimationSample object containing the image, mask, joints, areas, bounding boxes, and is_crowd.
        """
        image = cv2.imread(self.image_files[index])
        joints, bboxes = self.annotations[index]
        areas = np.array([w * h for (_, _, w, h) in bboxes], dtype=np.float32)
        iscrowd = np.zeros(len(joints), dtype=bool)
        mask = np.ones(image.shape[:2], dtype=np.float32)
        return PoseEstimationSample(image=image, mask=mask, joints=joints, areas=areas, bboxes_xywh=bboxes, is_crowd=iscrowd, additional_samples=None)
