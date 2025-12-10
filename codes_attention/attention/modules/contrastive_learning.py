import torch
from pytorch_lightning import LightningModule
from simclr.modules import NT_Xent
from modules import CLIP_Loss
from itertools import chain
from utils import get_logger
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np

class EEGContrastiveLearning(LightningModule):
    debug_logger = get_logger("dataloader_debug")

    def __init__(self, preprocess_dataset, args, encoder_raw_e, encoder_vocal, encoder_drum, encoder_bass, encoder_others, key):
        super().__init__()

        self.save_hyperparameters(args)

        self.encoder_raw_e = encoder_raw_e
        self.encoder_vocal = encoder_vocal
        self.encoder_drum = encoder_drum
        self.encoder_bass = encoder_bass
        self.encoder_others = encoder_others
        self.criterion = self.configure_criterion()
        self.attention_values = args.attention_values
        self.subject_id = args.subject_id
        self.song_id = args.song_id
        self.test_data_length=256
        self.audio_sample_rate=args.audio_sample_rate
        self.eeg_sample_rate=args.eeg_sample_rate

        self.key=key

        self.last_epoch_train_embeddings = []
        self.last_epoch_train_labels = []
        self.last_epoch_valid_embeddings = []
        self.last_epoch_valid_labels = []

        self.train_log_df = pd.DataFrame(
            columns=["Loss/train", "Accuracy/train_eeg", "Accuracy/train_audio"])
        self.valid_log_df = pd.DataFrame(
            columns=["Loss/valid", "Accuracy/valid_eeg", "Accuracy/valid_audio"])

        self.validation_end_values = []
        self.preprocess_dataset = preprocess_dataset
        self.matrix_list_all = []
        self.matrix_list_attention = []

        self.test_result = []
        self.label_accuracy_count = {
            label: {'correct': 0, 'total': 0} for label in range(10)}
        self.subject_accuracy_count = {
            subject: {'correct': 0, 'total': 0} for subject in range(24)}

    def forward(self, eeg, m_v, m_d, m_b, m_o):
        z_eeg = self.encoder_raw_e(eeg)
        z_v = self.encoder_vocal(m_v)
        z_d = self.encoder_drum(m_d)
        z_b = self.encoder_bass(m_b)
        z_o = self.encoder_others(m_o)

        return z_eeg, z_v, z_d, z_b, z_o

    def training_step(self, batch, batch_idx):
        eeg, m_v, m_d, m_b, m_o = batch[:5]
        task = batch[5]
        attention_score = batch[6]
        subject = batch[7]
        song = batch[8]
    
        z_eeg, z_v, z_d, z_b, z_o = self.forward(eeg, m_v, m_d, m_b, m_o)
        if self.hparams.detach_z_c:
            z_v = z_v.detach()
            z_d = z_d.detach()
            z_b = z_b.detach()
            z_o = z_o.detach()

        similarity_dict = self.criterion(
            z_eeg, z_v, z_d, z_b, z_o, task, attention_score,self.attention_values)
        if self.key == 'all':
            self.debug_logger.debug(f"Loss/train: {similarity_dict['all']['loss']}")
            loss = similarity_dict["all"]["loss"]
        elif self.key == 'high_attention':
            self.debug_logger.debug(f"Loss/train: {similarity_dict['attention']['loss']}")
            if similarity_dict["attention"]["loss"] is None:
                self.debug_logger.warning(f"Batch {batch_idx} has no valid loss, setting loss to 0.")
                loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            else:
                loss = similarity_dict["attention"]["loss"]
        else:
            raise ValueError('Please input training data category (all or high_attention)')


        if self.current_epoch == self.trainer.max_epochs - 1:
            self.last_epoch_train_embeddings.append(
                z_eeg.cpu().detach().numpy())

        return loss

    def on_validation_start(self):
        self.eval()
    def val_dataloader(self):
        return self.valid_loader
    def validation_step(self, batch, batch_idx):
        eeg, m_v, m_d, m_b, m_o = batch[:5]
        task = batch[5]
        attention_score = batch[6]
        subject = batch[7]
        song = batch[8]


        z_eeg, z_v, z_d, z_b, z_o = self.forward(eeg, m_v, m_d, m_b, m_o)
        if self.hparams.detach_z_c:
            z_v = z_v.detach()
            z_d = z_d.detach()
            z_b = z_b.detach()
            z_o = z_o.detach()
        similarity_dict = self.criterion(
            z_eeg, z_v, z_d, z_b, z_o, task, attention_score,self.attention_values)

        if self.key == 'all':
            self.debug_logger.info(f"Loss/train:{similarity_dict['all']['loss']}")
            loss = similarity_dict["all"]["loss"]
        elif self.key == 'high_attention':
            if similarity_dict["attention"]["loss"] is None:
                self.debug_logger.warning(f"Batch {batch_idx} has no valid loss, setting loss to 0.")
                loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            else:
                loss = similarity_dict["attention"]["loss"]
        else:
            raise ValueError('Please input training data category (all or high_attention)')
 
        self.debug_logger.info(
            f"Outputs for positive pairs:{similarity_dict['all']['positive_list']}")
        self.debug_logger.info(
            f"Outputs for negative pairs:{similarity_dict['all']['negative_av']}")
        self.debug_logger.info(
            f"Loss/valid_attention:{similarity_dict['attention']['loss']}")

        positive_list = similarity_dict["all"]["positive_list"]
        positive_task0 = self._get_tensor_value(positive_list[0])
        positive_task1 = self._get_tensor_value(positive_list[1])
        positive_task2 = self._get_tensor_value(positive_list[2])
        positive_task3 = self._get_tensor_value(positive_list[3])

        pos_list = [x for x in positive_list if x is not None]
        if len(pos_list) > 0:
            positive_average = sum(pos_list) / len(pos_list)



        filtered_list = similarity_dict["attention"]["positive_list"]
        filtered_task0 = self._get_tensor_value(filtered_list[0])
        filtered_task1 = self._get_tensor_value(filtered_list[1])
        filtered_task2 = self._get_tensor_value(filtered_list[2])
        filtered_task3 = self._get_tensor_value(filtered_list[3])

        fil_list = [x for x in filtered_list if x is not None]
        if len(fil_list) > 0:
            filtered_average = sum(fil_list) / len(fil_list)
        else:
            filtered_average = None

       
        matrix_all = similarity_dict["all"]["matrix_list"]
        matrix_attention = similarity_dict["attention"]["matrix_list"]

        self.matrix_list_all.append(matrix_all)
        self.matrix_list_attention.append(matrix_attention)
        
        evaluation_all = self.compute_evaluation_matrix(self.matrix_list_all)
        evaluation_attention = self.compute_evaluation_matrix(self.matrix_list_attention)

        print('Overall')
        print('大小関係', evaluation_all[0])
        print('大小4*1', evaluation_all[3])
        print('大小*1', evaluation_all[6])
        print('差分posグループ', evaluation_all[1])
        print('差分negグループ', evaluation_all[2])
        print('pos_4*1', evaluation_all[4])
        print('neg_4*1', evaluation_all[5])
        print('pos*1', evaluation_all[7])
        print('pos*1', evaluation_all[8])
        
        print('Attention')
        print('大小関係', evaluation_attention[0])
        print('大小4*1', evaluation_attention[3])
        print('大小*1', evaluation_attention[6])
        print('差分posグループ', evaluation_attention[1])
        print('差分negグループ', evaluation_attention[2])
        print('pos_4*1', evaluation_attention[4])
        print('neg_4*1', evaluation_attention[5])
        print('pos*1', evaluation_attention[7])
        print('neg*1', evaluation_attention[8])
        
        return loss


    def compute_evaluation_matrix(self, all_lists):
        if not all_lists or all(len(task) == 0 for group in all_lists for task in group):
            print("all_lists is empty")
            nan_matrix = torch.full((4, 4), float('nan'))
            nan_column = torch.full((4, 1), float('nan'))
            return [nan_matrix, nan_matrix, nan_matrix, nan_column, nan_column, nan_column,
                    float('nan'), float('nan'), float('nan')]

        first_nonempty = next((lists for lists in all_lists if any(len(t) > 0 for t in lists)), None)
        num_tasks = len(first_nonempty) if first_nonempty is not None else 4

        ratio_num_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
        ratio_den_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
        
        pos_sum_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
        pos_cnt_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
        neg_sum_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
        neg_cnt_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)

        ratio_num_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)
        ratio_den_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)
        
        pos_sum_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)
        pos_cnt_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)
        neg_sum_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)
        neg_cnt_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)

        total_number = 0
        larger_number = 0

        global_pos_sum = 0
        global_pos_cnt = 0
        global_neg_sum = 0
        global_neg_cnt = 0

        for lists in all_lists:
            for task_idx, task_list in enumerate(lists):
                if len(task_list) == 0:
                    continue

                task_tensor = torch.as_tensor(task_list, dtype=torch.float32)  
                pos_values = task_tensor[:, task_idx]  

                for neg_idx in range(num_tasks):
                    if task_idx == neg_idx:
                        continue
                    neg_values = task_tensor[:, neg_idx]

                    larger_count = (pos_values > neg_values).sum().item()
                    total_count = len(pos_values)
                    ratio_num_44[task_idx, neg_idx] += larger_count
                    ratio_den_44[task_idx, neg_idx] += total_count

                    differences = pos_values - neg_values
                    pos_sel = differences[differences > 0]
                    neg_sel = differences[differences < 0]
                    if len(pos_sel) > 0:
                        pos_sum_44[task_idx, neg_idx] += pos_sel.sum().item()
                        pos_cnt_44[task_idx, neg_idx] += len(pos_sel)
                    if len(neg_sel) > 0:
                        neg_sum_44[task_idx, neg_idx] += neg_sel.sum().item()
                        neg_cnt_44[task_idx, neg_idx] += len(neg_sel)

                max_values, _ = task_tensor[:, [i for i in range(num_tasks) if i != task_idx]].max(dim=1) 
                larger = (pos_values > max_values).sum().item()
                total = len(pos_values)
                total_number += total
                larger_number += larger

                ratio_num_41[task_idx, 0] += larger
                ratio_den_41[task_idx, 0] += total

                diffs = pos_values - max_values
                pos41 = diffs[diffs > 0]
                neg41 = diffs[diffs < 0]
                if len(pos41) > 0:
                    pos_sum_41[task_idx, 0] += pos41.sum().item()
                    pos_cnt_41[task_idx, 0] += len(pos41)
                    global_pos_sum += pos41.sum().item()
                    global_pos_cnt += len(pos41)
                if len(neg41) > 0:
                    neg_sum_41[task_idx, 0] += neg41.sum().item()
                    neg_cnt_41[task_idx, 0] += len(neg41)
                    global_neg_sum += neg41.sum().item()
                    global_neg_cnt += len(neg41)


        ratio_matrix = ratio_num_44/ratio_den_44  
        pos_diff     = pos_sum_44/pos_cnt_44    
        neg_diff     = neg_sum_44/neg_cnt_44    

        ratio_4      = ratio_num_41/ratio_den_41
        pos_diff_4   = pos_sum_41/pos_cnt_41
        neg_diff_4   = neg_sum_41/neg_cnt_41

        ratio = (larger_number / total_number) if total_number > 0 else float('nan')
        positive_difference = (global_pos_sum / global_pos_cnt) if global_pos_cnt > 0 else float('nan')
        negative_difference = (global_neg_sum / global_neg_cnt) if global_neg_cnt > 0 else float('nan')

        return [ratio_matrix, pos_diff, neg_diff,
                ratio_4, pos_diff_4, neg_diff_4,
                ratio, positive_difference, negative_difference]


    def test_step(self, batch, batch_idx):
        eeg, m_v, m_d, m_b, m_o = batch[:5]
        task = batch[5]
        attention_score = batch[6]
        subject = batch[7]
        song = batch[8]


        z_eeg, z_v, z_d, z_b, z_o = self.forward(eeg, m_v, m_d, m_b, m_o)
        if self.hparams.detach_z_c:
            z_v = z_v.detach()
            z_d = z_d.detach()
            z_b = z_b.detach()
            z_o = z_o.detach()
        similarity_dict = self.criterion(
            z_eeg, z_v, z_d, z_b, z_o, task, attention_score,self.attention_values)

        positive_list = similarity_dict["all"]["positive_list"]
        positive_task0 = self._get_tensor_value(positive_list[0])
        positive_task1 = self._get_tensor_value(positive_list[1])
        positive_task2 = self._get_tensor_value(positive_list[2])
        positive_task3 = self._get_tensor_value(positive_list[3])

        pos_list = [x for x in positive_list if x is not None]
        if len(pos_list) > 0:
            positive_average = sum(pos_list) / len(pos_list)

     
        filtered_list = similarity_dict["attention"]["positive_list"]
        filtered_task0 = self._get_tensor_value(filtered_list[0])
        filtered_task1 = self._get_tensor_value(filtered_list[1])
        filtered_task2 = self._get_tensor_value(filtered_list[2])
        filtered_task3 = self._get_tensor_value(filtered_list[3])

        fil_list = [x for x in filtered_list if x is not None]
        if len(fil_list) > 0:
            filtered_average = sum(fil_list) / len(fil_list)
        else:
            filtered_average = None

        matrix_all = similarity_dict["all"]["matrix_list"]
        matrix_attention = similarity_dict["attention"]["matrix_list"]

        self.matrix_list_all.append(matrix_all)
        self.matrix_list_attention.append(matrix_attention)
        
        evaluation_all = self.compute_evaluation_matrix(self.matrix_list_all)
        evaluation_attention = self.compute_evaluation_matrix(self.matrix_list_attention)

        print('Overall')
        print('大小関係', evaluation_all[0])
        print('大小4*1', evaluation_all[3])
        print('大小*1', evaluation_all[6])
        print('差分posグループ', evaluation_all[1])
        print('差分negグループ', evaluation_all[2])
        print('pos_4*1', evaluation_all[4])
        print('neg_4*1', evaluation_all[5])
        print('pos*1', evaluation_all[7])
        print('pos*1', evaluation_all[8])
        
        print('Attention')
        print('大小関係', evaluation_attention[0])
        print('大小4*1', evaluation_attention[3])
        print('大小*1', evaluation_attention[6])
        print('差分posグループ', evaluation_attention[1])
        print('差分negグループ', evaluation_attention[2])
        print('pos_4*1', evaluation_attention[4])
        print('neg_4*1', evaluation_attention[5])
        print('pos*1', evaluation_attention[7])
        print('neg*1', evaluation_attention[8])

    def on_test_end(self):
        return super().on_test_end()


    def configure_criterion(self):
        if self.hparams.accelerator == "dp" and self.hparams.gpus:
            batch_size = int(self.hparams.batch_size / self.hparams.gpus)
        else:
            batch_size = self.hparams.batch_size

        if self.hparams.loss_function == "clip_loss":
            print("use CLIP_loss as criterion")
            criterion = CLIP_Loss(
                batch_size, self.hparams.temperature, world_size=1)
        else:
            print('use NT_Xent as criterion')
            criterion = NT_Xent(
                batch_size, self.hparams.temperature, world_size=1)
        return criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.encoder_raw_e.parameters(), self.encoder_vocal.parameters(
        ), self.encoder_drum.parameters(), self.encoder_bass.parameters(), self.encoder_others.parameters()), self.hparams.learning_rate)
        return {"optimizer": optimizer}

    def _shared_step(self, label, y):
        y_hat = y
        y = label

        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (y == preds).sum() / y.size(0)
        return loss, acc

    def Kfold_log(self):
        return self.train_log_df, self.valid_log_df

    def save_checkpoint(self, filepath):
        torch.save({
            'module_state_dict': self.state_dict(),
            'encoder_raw_e_state_dict': self.encoder_raw_e.state_dict(),
            'encoder_vocal_state_dict': self.encoder_vocal.state_dict(),
            'encoder_drum_state_dict': self.encoder_drum.state_dict(),
            'encoder_bass_state_dict': self.encoder_bass.state_dict(),
            'encoder_others_state_dict': self.encoder_others.state_dict(),
            'optimizer_state_dict': self.trainer.optimizers[0].state_dict()
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['module_state_dict'])
        self.encoder_raw_e.load_state_dict(
            checkpoint['encoder_raw_e_state_dict'])
        self.encoder_vocal.load_state_dict(
            checkpoint['encoder_vocal_state_dict'])
        self.encoder_drum.load_state_dict(
            checkpoint['encoder_drum_state_dict'])
        self.encoder_bass.load_state_dict(
            checkpoint['encoder_bass_state_dict'])
        self.encoder_others.load_state_dict(
            checkpoint['encoder_others_state_dict'])

        optimizer_config = self.configure_optimizers()
        optimizer = optimizer_config['optimizer']

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return optimizer

    def _get_tensor_value(self, value):
        if value is None:
            return None
        return value.cpu().detach().numpy().item()

    def _get_eeg(self, eeg, eeg_start):
        slice_eeg = eeg[:, :, eeg_start: eeg_start +
                        self.test_data_length]

        return slice_eeg

    def _get_audio(self, audio, eeg_start):
        test_slice_length = int(
            self.test_data_length / self.eeg_sample_rate)
        audio_start = int(
            eeg_start * (self.audio_sample_rate / self.eeg_sample_rate))
        audio_end = int(audio_start + (test_slice_length *
                        self.audio_sample_rate))

        slice_audio = audio[:, :, audio_start:audio_end]

        return slice_audio



