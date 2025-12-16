import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from audiomentations import AddGaussianNoise, Gain
from datasets import get_dataset
from models import SampleCNN2DEEG
from modules import EEGContrastiveLearning
from utils import yaml_config_hook, get_logger, file_writer
from preprocessing import eeg_data_processing, experiment_data_processing
import pandas as pd
import datetime
import random
from pathlib import Path
import torch


def preprocess():
    input = config['raw_data_dir']
    output_base = f"{config['dataset_dir']}/eeg/"
    preprocess_logger = get_logger('preprocess')

    input_path = Path(f"{input}/eeg/")
    eeg_input_list = list(input_path.glob("*.csv"))
    if not eeg_input_list:
        preprocess_logger.error("Input eeg file does not exist.")
        return

    subject_num = 0
    subject_list = []
    for eeg_input in eeg_input_list:
        subject_list.append(
            {"subject_id": subject_num, "subject_name": eeg_input.stem})
        experiment_result = Path(f"{input}/experiment/{eeg_input.stem}.json")
        if not experiment_result.exists():
            preprocess_logger.error("Experiment result file does not exist.")
            return

        eeg_data_processing(eeg_input, experiment_result,
                            subject_num, output_base)
        subject_num += 1

    return subject_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PredANN")

    config = yaml_config_hook("/codes_attention/config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument('--mode', type=str)
    parser.add_argument('--start_position', type=int)
    parser.add_argument('--evaluation_length', type=int)
    parser.add_argument('--attention_values', type=int, nargs='+')
    parser.add_argument('--subject_id', type=int, default=None)
    parser.add_argument('--song_id', type=int, default=None)
    parser.add_argument('--key', type=str)
    parser.add_argument('--test_window_size', type=int)
    parser.add_argument('--test_stride', type=int)
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)

    if args.mode == "preprocess":
        result = preprocess()
        file_writer(f"{config['dataset_dir']}/preprocess_result.txt", result)

        exit()

    train_transform = {}
    if args.openmiir_augmentation == "gaussiannoise":
        train_transform = [
            AddGaussianNoise(min_amplitude=args.min_amplitude,
                             max_amplitude=args.max_amplitude, p=0.5),
        ]
        print("augematation is gaussiannoise")

    elif args.openmiir_augmentation == "gain":
        train_transform = [
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5)
        ]
        print("augematation is gain")

    elif args.openmiir_augmentation == "gaussiannoise+gain":
        train_transform = [
            AddGaussianNoise(min_amplitude=args.min_amplitude,
                             max_amplitude=args.max_amplitude, p=0.5),
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5)
        ]
        print("augematation is gaussiannoise+gain")

    else:
        print("no augmentation")

    train_log = pd.DataFrame(
        columns=["Loss/train", "Accuracy/train_eeg", "Accuracy/train_audio"])
    valid_log = pd.DataFrame(
        columns=["Loss/valid", "Accuracy/valid_eeg", "Accuracy/valid_audio"])

    train_dataset = get_dataset(
        args.dataset, args.dataset_dir, subset="train", download=False)
    train_dataset.set_sliding_window_parameters(args.window_size, args.stride)
    train_dataset.set_eeg_normalization(
        args.eeg_normalization, args.clamp_value)
    train_dataset.set_other_parameters( args.eeg_length, args.audio_clip_length, args.shifting_time, args.start_position)
    random.seed(42)
    train_random_numbers = [random.randint(
        0, args.eeg_sample_rate * 30 - args.eeg_length - 1) for _ in range(1200)]
    train_dataset.set_random_numbers(train_random_numbers)

    if args.openmiir_augmentation != "no_augmentation":
        train_dataset.set_transform(train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )

    valid_dataset = get_dataset(
        args.dataset, args.dataset_dir, subset="valid", download=False)
    valid_dataset.set_sliding_window_parameters(args.window_size, args.stride)
    valid_dataset.set_eeg_normalization(
        args.eeg_normalization, args.clamp_value)
    valid_dataset.set_other_parameters( args.eeg_length, args.audio_clip_length, args.shifting_time, args.start_position)
    random.seed(42)
    valid_random_numbers = [random.randint(
        0, args.window_size - args.eeg_length - 1) for _ in range(1200)]
    valid_dataset.set_random_numbers(valid_random_numbers)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False,
    )

    test_dataset = get_dataset(
        args.test_dataset, args.dataset_dir, subset="test", download=False)
    test_dataset.set_test_data_length(args.test_data_length)
    test_dataset.set_sliding_window_parameters(args.test_window_size, args.test_stride)
    test_dataset.set_eeg_normalization(
        args.eeg_normalization, args.clamp_value)
    test_dataset.set_other_parameters(
        args.eeg_length, args.audio_clip_length, args.shifting_time, args.start_position)
    random.seed(42)
    test_random_numbers = [random.randint(
        0, args.window_size - args.eeg_length - 1) for _ in range(1200)]
    test_dataset.set_random_numbers(test_random_numbers)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False,
    )

    print(f"Size of train dataset: {len(train_dataset)}")
    print(f"Size of valid dataset: {len(valid_dataset)}")
    print(f"Size of test dataset: {len(test_dataset)}")

    if args.dataset == "preprocessing_eegmusic":
        encoder_eeg = SampleCNN2DEEG(
            out_dim=train_dataset.labels(),
            kernal_size=3,
        )
        encoder_vocal = SampleCNN2DEEG(
            out_dim=train_dataset.labels(),
            kernal_size=3,
        )
        encoder_drum = SampleCNN2DEEG(
            out_dim=train_dataset.labels(),
            kernal_size=3,
        )
        encoder_bass = SampleCNN2DEEG(
            out_dim=train_dataset.labels(),
            kernal_size=3,
        )
        encoder_others = SampleCNN2DEEG(
            out_dim=train_dataset.labels(),
            kernal_size=3,
        )

    print('EEG Contrastive learning')
    module = EEGContrastiveLearning(
        valid_dataset, args, encoder_eeg, encoder_vocal, encoder_drum, encoder_bass, encoder_others,key=args.key)

    logger = TensorBoardLogger(
        "runs/{}".format(args.training_date), name="nmed-CL-{}".format(args.dataset))

    early_stop_callback = EarlyStopping(
        monitor="Valid/loss", patience=10
    )
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=args.max_epochs,
        deterministic=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        accelerator=args.accelerator,
        resume_from_checkpoint='/checkpoint_path',
        accumulate_grad_batches=6
    )
    print('[[[ START ]]]', datetime.datetime.now())
    checkpoint_path = "/checkpoint_path"
    checkpoint = torch.load(checkpoint_path)
    module.load_state_dict(checkpoint['state_dict'])
    trainer.test(module,dataloaders=test_loader)
    print('[[[ FINISH ]]]', datetime.datetime.now())

