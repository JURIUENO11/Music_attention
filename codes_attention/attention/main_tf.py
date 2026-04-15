import argparse
import numpy as np
import random
from pathlib import Path
import pandas as pd
from audiomentations import AddGaussianNoise, Gain
from datasets import get_dataset
from utils import yaml_config_hook, get_logger, file_writer
from preprocessing import eeg_data_processing
import tensorflow as tf
from models import AAD
import datetime
import mne
import librosa
import essentia
import essentia.standard as es
import torch

class EpochEndCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n[[[ Epoch {epoch + 1} finished ]]]")
def extract_audio_features(waveform, sr=44100, eeg_sr=256, n_mels=24):
    hop_size = sr // eeg_sr 
    frame_size = 276

    windowing = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    melbands = es.MelBands(numberBands=n_mels, inputSize=frame_size//2+1, sampleRate=sr)
    spectral_peaks = es.SpectralPeaks()   
    hpcp = es.HPCP(sampleRate=sr, harmonics=0,size=24)

    logmel_list, hpcp_list = [], []

    for frame in es.FrameGenerator(
        essentia.array(waveform),
        frameSize=frame_size,
        hopSize=hop_size,
        startFromZero=True
    ):
        windowed = windowing(frame)
        spec = spectrum(windowed)
        mel = melbands(spec)
        logmel = np.log(mel + 1e-8)
        freqs, mags = spectral_peaks(spec)
        hpcp_f = hpcp(mags, freqs)

        logmel_list.append(logmel)
        hpcp_list.append(hpcp_f)

    logmel_arr = np.stack(logmel_list, axis=0)
    hpcp_arr = np.stack(hpcp_list, axis=0)

    logmel_arr = logmel_arr[:768]
    hpcp_arr = hpcp_arr[:768]

    return logmel_arr.astype(np.float32), hpcp_arr.astype(np.float32)

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

    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument('--mode', type=str)
    parser.add_argument('--start_position', type=int)
    parser.add_argument('--evaluation_length', type=int)
    parser.add_argument('--attention_values', type=int, nargs='+')
    parser.add_argument('--subject_id', type=int, default=None)
    parser.add_argument('--song_id', type=int, default=None)
    parser.add_argument('--key', type=int)
    parser.add_argument('--test_window_size', type=int)
    parser.add_argument('--test_stride', type=int)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.mode == "preprocess":
        result = preprocess()
        file_writer(f"{config['dataset_dir']}/preprocess_result.txt", result)
        exit()

    train_transform = None
    if args.openmiir_augmentation == "gaussiannoise":
        train_transform = [
            AddGaussianNoise(min_amplitude=args.min_amplitude,
                             max_amplitude=args.max_amplitude, p=0.5)
        ]
        print("augmentation is gaussiannoise")
    elif args.openmiir_augmentation == "gain":
        train_transform = [
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5)
        ]
        print("augmentation is gain")
    elif args.openmiir_augmentation == "gaussiannoise+gain":
        train_transform = [
            AddGaussianNoise(min_amplitude=args.min_amplitude,
                             max_amplitude=args.max_amplitude, p=0.5),
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5)
        ]
        print("augmentation is gaussiannoise+gain")
    else:
        print("no augmentation")

    train_dataset = get_dataset(
        args.dataset, args.dataset_dir, subset="train", download=False)
    train_dataset.set_sliding_window_parameters(args.window_size, args.stride)
    train_dataset.set_eeg_normalization(args.eeg_normalization, args.clamp_value)
    train_dataset.set_other_parameters(args.eeg_length, args.audio_clip_length, args.shifting_time, args.start_position)
    random.seed(42)
    train_random_numbers = [random.randint(
        0, args.eeg_sample_rate * 30 - args.eeg_length - 1) for _ in range(1200)]
    train_dataset.set_random_numbers(train_random_numbers)

    if train_transform:
        train_dataset.set_transform(train_transform)

    valid_dataset = get_dataset(
        args.dataset, args.dataset_dir, subset="valid", download=False)
    valid_dataset.set_sliding_window_parameters(args.window_size, args.stride)
    valid_dataset.set_eeg_normalization(args.eeg_normalization, args.clamp_value)
    valid_dataset.set_other_parameters(args.eeg_length, args.audio_clip_length, args.shifting_time, args.start_position)
    random.seed(42)
    valid_random_numbers = [random.randint(
        0, args.window_size - args.eeg_length - 1) for _ in range(1200)]
    valid_dataset.set_random_numbers(valid_random_numbers)

    eegs = []
    labels = []
    print("[[[ Loading all train EEG for CSP fit ]]]")
    for i in range(len(train_dataset)):
        e, aa0, aa1, aa2, aa3, task, attention_score, subject, song = train_dataset[i]
        eegs.append(e.numpy())
        labels.append(task)

    eegs_np = np.stack(eegs).astype('float64')
    labels_np = np.array(labels)

    csp = mne.decoding.CSP(
        n_components=eegs_np.shape[1],
        transform_into='csp_space',
        reg='oas'
    )
    print("[[[ Fitting CSP ]]]")
    csp.fit(eegs_np, labels_np)

    del eegs, labels, eegs_np, labels_np
    print("[[[ CSP finished ]]]")

    # Generator
    def generator(dataset):
        for i in range(len(dataset)):
            e, a0, a1, a2, a3, label, *_ = dataset[i]

            a0 = a0.mean(axis=0)  
            a1 = a1.mean(axis=0)
            a2 = a2.mean(axis=0)
            a3 = a3.mean(axis=0)

            e_csp = csp.transform(np.expand_dims(e.numpy(), axis=0)).astype(np.float32)[0]
            e_csp = np.transpose(e_csp, (1,0))

            # audio features
            a0_logmel, a0_hpcp = extract_audio_features(a0.numpy())
            fused0 = np.stack([a0_logmel, a0_hpcp], axis=-1)
            a1_logmel, a1_hpcp = extract_audio_features(a1.numpy())
            fused1 = np.stack([a1_logmel, a1_hpcp], axis=-1)
            a2_logmel, a2_hpcp = extract_audio_features(a2.numpy())
            fused2 = np.stack([a2_logmel, a2_hpcp], axis=-1)
            a3_logmel, a3_hpcp = extract_audio_features(a3.numpy())
            fused3 = np.stack([a3_logmel, a3_hpcp], axis=-1)

            yield (
                (
                    tf.convert_to_tensor(e_csp),
                    tf.convert_to_tensor(fused0),
                    tf.convert_to_tensor(fused1),
                    tf.convert_to_tensor(fused2),
                    tf.convert_to_tensor(fused3)
                ),
                tf.convert_to_tensor(label)
            )

    output_signature=(
        (
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),        
            tf.TensorSpec(shape=(None,24,2), dtype=tf.float32),        
            tf.TensorSpec(shape=(None,24,2), dtype=tf.float32),         
            tf.TensorSpec(shape=(None,24,2), dtype=tf.float32),        
            tf.TensorSpec(shape=(None,24,2), dtype=tf.float32),        
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    train_tf_dataset = tf.data.Dataset.from_generator(
        lambda: generator(train_dataset),
        output_signature=output_signature
    ).shuffle(1024).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    valid_tf_dataset = tf.data.Dataset.from_generator(
        lambda: generator(valid_dataset),
        output_signature=output_signature
    ).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # --------------------------------------------------------
    example_batch = next(iter(train_tf_dataset))
    (eeg_ex, fused0_ex, fused1_ex, fused2_ex, fused3_ex), label_ex = example_batch

    model = AAD(
        shape_eeg=eeg_ex.shape[1:],     
        shape_sti=fused0_ex.shape[1:] 
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )

    log_dir = f"runs/{args.training_date}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    print('[[[ START TRAINING ]]]', datetime.datetime.now())
    model.fit(
        train_tf_dataset,
        validation_data=valid_tf_dataset,
        epochs=args.max_epochs,
        callbacks=[
            tensorboard_callback,
            early_stop,
            tf.keras.callbacks.ModelCheckpoint(
                filepath="previous_paper_model_all.ckpt",
                save_best_only=False,
                save_weights_only=True
            ),
            EpochEndCallback()]
    )
    print('[[[ FINISH TRAINING ]]]', datetime.datetime.now())