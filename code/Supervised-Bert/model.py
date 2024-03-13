# basic
import os
import sys
import json
import shutil
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# torch
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

# from torchmetrics.classification import MulticlassAccuracy

# pl
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    DeviceStatsMonitor,
    StochasticWeightAveraging,
)

# transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from func import get_scheduler_batch, get_pool_emb, init_logger

# env
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ## Dataset and Dataloader


class Dataset_medical_exam(Dataset):
    def __init__(self, list_data, list_label, tokenizer, max_length_token):
        self.data = list_data
        self.label = list_label
        self.tokenizer = tokenizer
        self.max_length_token = max_length_token
        assert len(self.data) == len(self.label)

    def __getitem__(self, index):
        string_question_option = self.data[index]
        label_answer = self.label[index]
        inputs = self.tokenizer(
            string_question_option,
            padding="max_length",
            truncation=True,
            max_length=self.max_length_token,
            return_tensors="pt",
        )
        return (
            inputs["input_ids"].squeeze(),
            inputs["attention_mask"].squeeze(),
            label_answer,
        )

    def __len__(self):
        return len(self.data)


class Dataloader_medical_exam(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.logger = args.logger
        self.tokenizer = tokenizer
        self.max_length_token = args.max_length_token
        self.batchsize_train = args.batchsize_train
        self.batchsize_valid = args.batchsize_valid
        self.num_workers = 8
        self.dir_data = "data/{}.json"

    def format_data(self, list_dict_data):
        list_data, list_label = [], []
        for dict_data in list_dict_data:
            # requirement: five options and single answer
            if (
                not isinstance(dict_data["answer"], str)
                or len(dict_data["answer"]) > 1
                or not dict_data.get("E")
            ):
                continue
            string_question_option = (
                dict_data["question"]
                + " #"
                + dict_data["A"]
                + " #"
                + dict_data["B"]
                + " #"
                + dict_data["C"]
                + " #"
                + dict_data["D"]
                + " #"
                + dict_data["E"]
            )
            answer_label = ord(dict_data["answer"]) - ord("A")
            list_data.append(string_question_option)
            list_label.append(answer_label)
        return list_data, list_label

    def prepare_data(self):
        with open(self.dir_data.format("CMExam-train-aug"), "r", encoding="utf-8") as f:
            list_dict_train = json.load(f)
        with open(self.dir_data.format("CMExam-val"), "r", encoding="utf-8") as f:
            list_dict_valid = json.load(f)
        with open(self.dir_data.format("CNMLE-2022"), "r", encoding="utf-8") as f:
            list_dict_test = json.load(f)
        data_train, label_train = self.format_data(list_dict_train)
        data_valid, label_valid = self.format_data(list_dict_valid)
        data_test, label_test = self.format_data(list_dict_test)
        self.logger.info(
            f"Training Size: {len(data_train)}, Evaluation Size: {len(data_valid)}, Testing Size: {len(data_test)}"
        )
        # for idx in random.choices(list(range(len(data_train))), k=5):
        for idx in [0, 1, 2]:
            self.logger.info(f"Training Sample: {data_train[idx]}")
            self.logger.info(f"Training Label: {label_train[idx]}")

        return data_train, label_train, data_valid, label_valid, data_test, label_test

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        (
            data_train,
            label_train,
            data_valid,
            label_valid,
            data_test,
            label_test,
        ) = self.prepare_data()

        # dataset of each part
        self.dataset_train = Dataset_medical_exam(
            list_data=data_train,
            list_label=label_train,
            tokenizer=self.tokenizer,
            max_length_token=self.max_length_token,
        )
        self.dataset_valid = Dataset_medical_exam(
            list_data=data_valid,
            list_label=label_valid,
            tokenizer=self.tokenizer,
            max_length_token=self.max_length_token,
        )
        self.dataset_test = Dataset_medical_exam(
            list_data=data_test,
            list_label=label_test,
            tokenizer=self.tokenizer,
            max_length_token=self.max_length_token,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batchsize_train,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_valid,
            batch_size=self.batchsize_valid,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batchsize_valid,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )


# ## Trainer


class medical_exam(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        # model
        self.path_model = self.args.path_model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.path_model, num_labels=args.num_label, return_dict=True
        )
        # loss
        self.criterion_loss = nn.CrossEntropyLoss()
        # metric
        self.metric_acc = Accuracy()

    # model out
    def forward(self, batch_input, batch_mask):
        out = self.model(input_ids=batch_input, attention_mask=batch_mask)
        logits = out["logits"]
        return logits

    # step template
    def step(self, batch, batch_idx):
        batch_input, batch_mask, batch_label = batch
        batch_pred = self(batch_input, batch_mask)
        batch_label = batch_label.squeeze()
        # loss
        loss = self.criterion_loss(batch_pred, batch_label)
        return {
            "loss": loss,
            "batch_pred": batch_pred.detach(),
            "batch_label": batch_label.detach(),
        }

    # metrics template
    def step_metric(self, pred, target, label):
        step_acc = self.metric_acc(pred, target).item()
        self.log(f"{label}_acc", step_acc, prog_bar=True, sync_dist=True)

    # step_end template
    def step_end(self, outputs, label):
        # metrics
        if label != "train":
            self.step_metric(outputs["batch_pred"], outputs["batch_label"], label)
        # loss
        if label != "test":
            self.log(
                f"{label}_loss",
                outputs["loss"].detach().item(),
                prog_bar=False,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def training_step_end(self, outputs):
        self.step_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step_end(self, outputs):
        self.step_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step_end(self, outputs):
        self.step_end(outputs, "test")

    def predict_step(self, batch, batch_idx):
        result = self.test_step(batch, batch_idx)
        return result

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        # filter(lambda p: p.requires_grad, model_clf.parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        # torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)
        # No scheduler
        if self.args.scheduler == None:
            print(f"No scheduler and fixed lr={str(self.args.learning_rate)}")
            return optimizer
        else:
            scheduler = get_scheduler_batch(
                optimizer,
                self.args.scheduler,
                lr_base=self.args.learning_rate,
                num_epoch=self.args.epoch_max,
                num_batch=self.args.num_batch_train,
            )
            print(
                f"scheduler:{self.args.scheduler} and lr={str(self.args.learning_rate)}"
            )
            return ([optimizer], [{"scheduler": scheduler, "interval": "step"}])

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        return parser
        # pass


# ## Model


# class Bert_base(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.config = args
#         #  bert model
#         self.bert = AutoModel.from_pretrained(
#             self.config.path_model, return_dict=True, output_hidden_states=True
#         )
#         self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
#         self.classifier = nn.Linear(
#             in_features=self.config.dim_emb, out_features=self.config.dim_out
#         )
#         logger.info("Init BertNote")

#     def forward(self, batch_input, batch_mask, batch_label):
#         batch_emb = self.bert(input_ids=batch_input, attention_mask=batch_mask)
#         batch_emb = get_pool_emb(
#             batch_emb["hidden_states"], batch_mask, pool_strategy="cls"
#         )
#         # dropout
#         batch_emb = self.dropout(batch_emb)
#         # final output
#         batch_logits = self.classifier(batch_emb)
#         return {"logits": batch_logits}

MODEL_PATH = {
    # self
    "Bert_base": "path_of_model",
    "Roberta_base": "path_of_model",
    "Roberta_large": "path_of_model",
    # "Roberta_large": "/home/wjg/PLMs/chinese-roberta-wwm-ext-large",
    "mc_Bert_base": "path_of_model",
}


class argparse:
    pass


def main(args, logger):
    # seed
    pl.seed_everything(args.seed)

    # initiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.path_model)

    # loading data
    data = Dataloader_medical_exam(args, tokenizer)
    data.setup()

    args.num_batch_train = len(data.train_dataloader()) // len(args.devices) + 1
    logger.info(
        f"Training: num_epoch:{args.epoch_max}, num_batch:{args.num_batch_train}, all:{args.epoch_max*args.num_batch_train}"
    )

    # initiate model
    model_medical_exam = medical_exam(args=args)

    # callbacks
    checkpoint = ModelCheckpoint(
        filename="{epoch:02d}-{val_acc:.4f}",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        monitor=args.metric,
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor=args.metric, patience=args.earlystop_patience, mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, early_stopping, lr_monitor]

    logger_tb = TensorBoardLogger(args.dir_save, name="TensorBoardLogger")

    logger.info(f"hparams.auto_lr_find={args.auto_lr_find}")
    if args.auto_lr_find:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[0],
            # callbacks=callbacks,
            max_epochs=args.epoch_max,
            min_epochs=args.epoch_min,
            val_check_interval=args.val_check_interval,
            gradient_clip_val=args.gradient_clip_val,
            deterministic=True,
            logger=logger_tb,
            log_every_n_steps=10,
            # profiler="simple",
        )
        # learning rate
        lr_finder = trainer.tuner.lr_find(
            model_medical_exam,
            datamodule=data,
        )
        fig = lr_finder.plot(suggest=True)
        plt.savefig(args.dir_save + "/lr_find.svg", dpi=300)
        lr = lr_finder.suggestion()
        logger.info("suggest lr=", lr)
        model_medical_exam.hparams.learning_rate = lr
        args.learning_rate = lr
        del trainer
        del model_medical_exam
        model_medical_exam = model_medical_exam(args, learning_rate=args.learning_rate)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp_find_unused_parameters_false",
        callbacks=callbacks,
        max_epochs=args.epoch_max,
        min_epochs=args.epoch_min,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=True,
        logger=logger_tb,
        log_every_n_steps=1,
        # profiler="simple",
    )

    trainer.fit(model_medical_exam, data)

    logger.info(f"best_model_path: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"best_model_score: {trainer.checkpoint_callback.best_model_score}")

    logger.info(f"Now, let's see the testing result!!!")
    test_result = trainer.test(
        model_medical_exam, data.test_dataloader(), ckpt_path="best", verbose=True
    )
    logger.info(test_result)
    # save
    model_medical_exam.model.save_pretrained(args.dir_model_save)
    tokenizer.save_pretrained(args.dir_model_save)


if __name__ == "__main__":
    # mode
    mode = sys.argv[1]
    args = argparse()
    args.seed = 42
    # bert model
    args.mode = mode
    args.path_model = MODEL_PATH[args.mode]
    args.max_length_token = 256

    # run
    args.epoch_max = 10
    args.epoch_min = 6
    args.step_accumulate = 1
    args.val_check_interval = 0.2
    args.gradient_clip_val = 1.0
    args.earlystop_patience = 20
    args.batchsize_train = 32
    args.batchsize_valid = 300
    # learning
    args.learning_rate = 2e-5
    args.weight_decay = 0.01
    args.adam_epsilon = 1e-8
    args.auto_lr_find = False
    args.scheduler = "CyclicLR"
    # model
    args.metric = "val_acc"
    args.num_label = 5
    # args.dim_emb = 768
    # args.dim_out = 5
    # args.hidden_dropout_prob = 0.1

    # devide
    args.devices = [4, 5, 6, 7]
    # [6, 7]
    # [4, 5]
    # [0, 1, 2, 3]
    # [4, 5, 6, 7]
    # ,1,2,3,4,5,6,7

    # get the record of save
    path_dir_record = "log"
    args.dir_save = os.path.join(path_dir_record, args.mode)
    if not os.path.exists(args.dir_save):
        os.makedirs(args.dir_save)
    time_now = str(datetime.datetime.now().strftime("%d-%H:%M:%S"))
    time_now = args.mode + "-" + str(args.seed) + "-" + time_now
    args.dir_save = os.path.join(args.dir_save, time_now)
    if not os.path.exists(args.dir_save):
        os.makedirs(args.dir_save)
    args.dir_model_save = os.path.join(args.dir_save, args.mode)

    logger = init_logger(os.path.join(args.dir_save, f"{args.mode}.log"))
    args.logger = logger

    # shutil.copy("run.sh", os.path.join(args.dir_save, f"run_backup.sh"))
    # logger.info(
    #     f"run.sh has been backup in {os.path.join(args.dir_save, f'run_backup.sh')}"
    # )

    for key, value in args.__dict__.items():
        logger.info(f"{key}: {value}")

    main(args, logger)

    path_file_out = f"out/{args.mode}.out"
    if os.path.exists(path_file_out):
        shutil.copy(f"{path_file_out}", os.path.join(args.dir_save, path_file_out[4:]))
        logger.info(f"out backup in {os.path.join(args.dir_save, path_file_out[4:])}")

    logger.info("All done")
