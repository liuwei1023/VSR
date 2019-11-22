import json
import os 
import numpy as np 

class Config:
    def __init__(self):
        self._configs = {}

        # net
        self._configs["net"] = "EDVR"

        # WDSR_A
        self._configs["n_resblocks"] = 16

        # data
        self._configs["input_size"] = [256,256]
        self._configs["depth"] = 9

        # dir
        self._configs["result_path"] = "train_result"
        self._configs["best_model_name"] = "best.pkl"
        self._configs["min_loss_model_name"] = "min_loss.pkl"
        self._configs["ckpt_name"] = "ckpt_epoch_{}.pkl"

        # training
        self._configs["batch_size"] = 16
        self._configs["loss"] = "L1"
        self._configs["PP_loss"] = False
        self._configs["max_epochs"] = 50
        self._configs["lr"] = 0.0001
        self._configs["learning_mode"] = "cosin"
        self._configs["multi_step"] = [20]
        self._configs["validation_per_epochs"] = 10
        self._configs["validation_batch_size"] = 16

        # augment
        self._configs["flip"] = True

        # dataset
        self._configs["seg_frame"] = False
        self._configs["extension"] = "png"
        self._configs["MiniTest"] = False

        # non-local block
        self._configs["non_local"] = [1, 3, 2]

    @property
    def net(self):
        return self._configs["net"]

    @property
    def n_resblocks(self):
        return self._configs["n_resblocks"]

    @property
    def input_size(self):
        return self._configs["input_size"]

    @property
    def depth(self):
        return self._configs["depth"]

    @property
    def batch_size(self):
        return self._configs["batch_size"]

    @property
    def loss(self):
        return self._configs["loss"]

    @property
    def PP_loss(self):
        return self._configs["PP_loss"]

    @property
    def max_epochs(self):
        return self._configs["max_epochs"]

    @property
    def validation_per_epochs(self):
        return self._configs["validation_per_epochs"]

    @property
    def validation_batch_size(self):
        return self._configs["validation_batch_size"]

    @property
    def lr(self):
        return self._configs["lr"]

    @property
    def learning_mode(self):
        return self._configs["learning_mode"]
    
    @property
    def multi_step(self):
        return self._configs["multi_step"]

    @property
    def non_local(self):
        return self._configs["non_local"]

    @property
    def extension(self):
        return self._configs["extension"]

    @property
    def MiniTest(self):
        return self._configs["MiniTest"]

    @property
    def seg_frame(self):
        return self._configs["seg_frame"]

    @property
    def result_path(self):
        rPath = self._configs["result_path"]
        rPath = f"{rPath}_{self.extension}"
        if self.seg_frame:
            rPath = f"{rPath}_seg"

        if not os.path.exists(rPath):
            os.makedirs(rPath)
        return rPath

    @property
    def best_model_path(self):
        best_model_name = self._configs["best_model_name"]
        bPath = os.path.join(self.result_path, best_model_name)
        return bPath

    @property
    def min_loss_model_path(self):
        min_loss_model_name = self._configs["min_loss_model_name"]
        mPath = os.path.join(self.result_path, min_loss_model_name)
        return mPath

    @property
    def ckpt_path(self):
        ckpt_name = self._configs["ckpt_name"]
        cPath = os.path.join(self.result_path, ckpt_name)
        return cPath

    @property
    def flip(self):
        return self._configs["flip"]

    @property
    def log_path(self):
        lPath = os.path.join(self.result_path, "log.log")
        return lPath
    
    def config_all(self):
        return self._configs

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key] 

system_config = Config()