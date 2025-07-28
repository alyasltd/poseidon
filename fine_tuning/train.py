import torch

from runway_pose_dataset import RunwayPoseEstimationDataset # Import the custom dataset class

from torch.utils.data import DataLoader
from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer

from super_gradients.training.models.pose_estimation_models.yolo_nas_pose import YoloNASPosePostPredictionCallback
from super_gradients.training.utils.callbacks import ExtremeBatchPoseEstimationVisualizationCallback, Phase
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.metrics import PoseEstimationMetrics
from super_gradients.training.datasets.pose_estimation_datasets import YoloNASPoseCollateFN

from super_gradients.training.transforms.keypoints import (
    KeypointsRandomHorizontalFlip,
    KeypointsHSV,
    KeypointsBrightnessContrast,
    KeypointsRandomAffineTransform,
    KeypointsLongestMaxSize,
    KeypointsPadIfNeeded,
    KeypointsImageStandardize,
    KeypointsRemoveSmallObjects,
)


#After creating the yolonaspose base, we initialize the dataset and split it into training and validation sets.
RunwayPoseEstimationDataset.split_runway_pose_dataset(
    annotation_file="/home/aws_install/data/yolonas_pose_base/annotations/yolonas_pose_annotations.json",
    train_annotation_file="/home/aws_install/data/yolonas_pose_base/annotations/yolonas_pose_train.json",
    val_annotation_file="/home/aws_install/data/yolonas_pose_base/annotations/yolonas_pose_val.json",
    test_annotation_file="/home/aws_install/data/yolonas_pose_base/annotations/yolonas_pose_test.json",
    test_fraction=0.2,  # 20% for test
    val_fraction=0.2
)

# we define key parameters for the pose estimation task such as
# keypoint names, flip indexes, edge links, and colors.
KEYPOINT_NAMES = [
    "corner_top_left",
    "corner_top_right",
    "corner_bottom_right",
    "corner_bottom_left",
]

FLIP_INDEXES = [
    1, 0, 3, 2
]

EDGE_LINKS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
]

EDGE_COLORS = [[0, 255, 0]] * len(EDGE_LINKS)
KEYPOINT_COLORS = [[255, 0, 0]] * len(KEYPOINT_NAMES)
NUM_JOINTS = len(KEYPOINT_NAMES)
#print(f"Number of joints: {NUM_JOINTS}")
# OKS_SIGMAS = [0.07] * NUM_JOINTS
# → Each joint has the same standard deviation (σ) of 0.07.
# → This defines how tolerant the OKS metric is to localization error for each joint.
# → When multiplied by 4, it's scaling sigma to pixel space (to align with image size).
# → OKS is like IoU, but for keypoints: it scores how close predicted joints are.
OKS_SIGMAS = [0.07] * NUM_JOINTS 


#we apply the transforms to the training and validation datasets.

IMAGE_SIZE = 640

train_transforms = [
    KeypointsRandomHorizontalFlip(flip_index=FLIP_INDEXES, prob=0.5), # Random horizontal flip with specified keypoint flip indexes
    KeypointsHSV(prob=0.5, hgain=20, sgain=20, vgain=20), # Random HSV adjustments
    KeypointsBrightnessContrast(prob=0.5, brightness_range=[0.8, 1.2], contrast_range=[0.8, 1.2]), # Random brightness and contrast adjustments
    KeypointsRandomAffineTransform(
        max_rotation=15,  # Maximum rotation in degrees
        min_scale=0.8,  # Minimum scale factor
        max_scale=1.2, # Maximum scale factor
        max_translate=0.1, # Maximum translation as a fraction of the image size
        image_pad_value=127,    # Padding value for the image
        mask_pad_value=1, # Padding value for the mask
        prob=0.75, # Probability of applying the affine transformation
        interpolation_mode=[0, 1, 2, 3, 4], # Interpolation modes to choose from
    ),
    KeypointsLongestMaxSize(max_height=IMAGE_SIZE, max_width=IMAGE_SIZE), # Resize the image to the longest side with a maximum size
    KeypointsPadIfNeeded( 
        min_height=IMAGE_SIZE, # Minimum height after padding
        min_width=IMAGE_SIZE, # Minimum width after padding
        image_pad_value=[127, 127, 127], # Padding value for the image
        mask_pad_value=1, # Padding value for the mask
        padding_mode="bottom_right", # Padding mode to use
    ),
    KeypointsImageStandardize(max_value=255),
    KeypointsRemoveSmallObjects(min_instance_area=1, min_visible_keypoints=1),
]

val_transforms = [
    KeypointsLongestMaxSize(max_height=IMAGE_SIZE, max_width=IMAGE_SIZE),
    KeypointsPadIfNeeded(
        min_height=IMAGE_SIZE,
        min_width=IMAGE_SIZE,
        image_pad_value=[127, 127, 127],
        mask_pad_value=1,
        padding_mode="bottom_right",
    ),
    KeypointsImageStandardize(max_value=255),
]

# Create the training and validation datasets using the custom dataset class
train_dataset = RunwayPoseEstimationDataset(
    data_dir="/home/aws_install/data/yolonas_pose_base",     # Root directory of the dataset
    images_dir="/home/aws_install/data/yolonas_pose_base/images",
    json_file="/home/aws_install/data/yolonas_pose_base/annotations/yolonas_pose_train.json",
    transforms=train_transforms, 
    edge_links=EDGE_LINKS,
    edge_colors=EDGE_COLORS,
    keypoint_colors=KEYPOINT_COLORS,
)

val_dataset = RunwayPoseEstimationDataset(
    data_dir="/home/aws_install/data/yolonas_pose_base",     # Root directory of the dataset
    images_dir="/home/aws_install/data/yolonas_pose_base/images",
    json_file="/home/aws_install/data/yolonas_pose_base/annotations/yolonas_pose_val.json",
    transforms=val_transforms,
    edge_links=EDGE_LINKS,
    edge_colors=EDGE_COLORS,
    keypoint_colors=KEYPOINT_COLORS,
)

test_dataset = RunwayPoseEstimationDataset(
    data_dir="/home/aws_install/data/yolonas_pose_base",     # Root directory of the dataset
    images_dir="/home/aws_install/data/yolonas_pose_base/images",
    json_file="/home/aws_install/data/yolonas_pose_base/annotations/yolonas_pose_test.json",
    transforms=val_transforms,
    edge_links=EDGE_LINKS,
    edge_colors=EDGE_COLORS,
    keypoint_colors=KEYPOINT_COLORS,
)

# Create dataloaders
train_dataloader_params = {"shuffle": True, "batch_size": 24, "drop_last": True, "pin_memory": False, "collate_fn": YoloNASPoseCollateFN()}

val_dataloader_params = {"shuffle": True, "batch_size": 24, "drop_last": True, "pin_memory": False, "collate_fn": YoloNASPoseCollateFN()}

train_dataloader = DataLoader(train_dataset, **train_dataloader_params)

val_dataloader = DataLoader(val_dataset, **val_dataloader_params)

test_dataloader = DataLoader(test_dataset, **val_dataloader_params)


# Set the device, TRainer and model for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "checkpoints"
trainer = Trainer(experiment_name="lard_ft_S_100_w_test", ckpt_root_dir=CHECKPOINT_DIR)

yolo_nas_pose = models.get(
    Models.YOLO_NAS_POSE_S,
    num_classes=NUM_JOINTS
).to(device)


# Define the post-prediction callback for pose estimation
post_prediction_callback = YoloNASPosePostPredictionCallback(
    pose_confidence_threshold=0.01,
    nms_iou_threshold=0.7,
    pre_nms_max_predictions=300,
    post_nms_max_predictions=30,
)

metrics = PoseEstimationMetrics(
    num_joints=NUM_JOINTS,
    oks_sigmas=OKS_SIGMAS,
    max_objects_per_image= 1,
    post_prediction_callback=post_prediction_callback,
    iou_thresholds_to_report=[0.5, 0.75]
)

visualization_callback = ExtremeBatchPoseEstimationVisualizationCallback(
    keypoint_colors=KEYPOINT_COLORS,
    edge_colors=EDGE_COLORS,
    edge_links=EDGE_LINKS,
    loss_to_monitor="YoloNASPoseLoss/loss",
    max=True,
    freq=1,
    max_images=16,
    enable_on_train_loader=True,
    enable_on_valid_loader=True,
    post_prediction_callback=post_prediction_callback,
)

early_stop = EarlyStop(
    phase=Phase.VALIDATION_EPOCH_END,
    monitor="AP",
    mode="max",
    min_delta=0.0001,
    patience=100,
    verbose=True,
)

train_params = {
    "warmup_mode": "LinearBatchLRWarmup",
    "warmup_initial_lr": 1e-8,
    "lr_warmup_epochs": 2,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.05,
    "max_epochs": 200,
    "zero_weight_decay_on_bias_and_bn": True,
    "batch_accumulate": 1,
    "average_best_models": True,
    "save_ckpt_epoch_list": [],
    "loss": "yolo_nas_pose_loss",
    "criterion_params": {
        "oks_sigmas": OKS_SIGMAS,
        "classification_loss_weight": 1.0,
        "classification_loss_type": "focal",
        "regression_iou_loss_type": "ciou",
        "iou_loss_weight": 2.5,
        "dfl_loss_weight": 0.01,
        "pose_cls_loss_weight": 1.0,
        "pose_reg_loss_weight": 34.0,
        "pose_classification_loss_type": "focal",
        "rescale_pose_loss_with_assigned_score": True,
        "assigner_multiply_by_pose_oks": True,
    },
    "optimizer": "AdamW",
    "optimizer_params": {"weight_decay": 0.000001},
    "ema": True,
    "ema_params": {"decay": 0.997, "decay_type": "threshold"},
    "mixed_precision": True, #technically autocast should be enabled but not the case here
    "sync_bn": False,
    "valid_metrics_list": [metrics],
    "phase_callbacks": [visualization_callback, early_stop],
    "pre_prediction_callback": None,
    "metric_to_watch": "AP",
    "greater_metric_to_watch_is_better": True,
}


# Note, this is training for 100 epochs 
trainer.train(model=yolo_nas_pose, training_params=train_params, train_loader=train_dataloader, valid_loader=val_dataloader)


post_prediction_callback = YoloNASPosePostPredictionCallback(
    pose_confidence_threshold=0.01,
    nms_iou_threshold=0.7,
    pre_nms_max_predictions=300,
    post_nms_max_predictions=30,
)

metrics = PoseEstimationMetrics(
    num_joints=NUM_JOINTS,
    oks_sigmas=OKS_SIGMAS,
    max_objects_per_image=30,
    post_prediction_callback=post_prediction_callback,
)
best_model = models.get("yolo_nas_pose_s", num_classes=NUM_JOINTS, checkpoint_path="/home/aws_install/poseidon/fine_tuning/checkpoints/lard_ft_S_100/RUN_20250709_140953_233014/ckpt_best.pth")
     
trainer.test(model=best_model, test_loader=test_dataloader, test_metrics_list=metrics)

img_url = "/home/aws_install/data/yolonas_pose_base/images/test/ZBAA_36R_35_29.jpeg"
best_model.predict(img_url, conf=0.20, fuse_model=False).show()