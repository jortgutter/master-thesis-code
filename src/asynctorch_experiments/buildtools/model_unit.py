# asynctorch_experiment imports
from asynctorch_experiments.buildtools.parameters import BuildParams, VersionedObject
from asynctorch_experiments.buildtools.better_paths import MyPath
from asynctorch_experiments.traintools.model_trainer import ModelTrainer
from asynctorch_experiments.evaluation.monitor import Monitor, TestMonitor, SpikeMonitor, MembraneMonitor
import numpy as np

class MetaFile(VersionedObject):
    meta_file_version='0.1.0'
    def __init__(self, file_path, epoch):
        super().__init__(version=MetaFile.meta_file_version)
        self.file_path = file_path
        self.was_trained=False
        self.train_accs={}
        self.train_losses={}
        self.test_accs={}
        self.test_losses={}
        self.epoch=epoch
        

    def exists(self):
        return self.file_path.exists()
        
    def save(self):
        self.file_path.save(
            file_contents=self
        )
        
    def check_F_test(self, F_test):
        return F_test in self.test_accs
    
    def check_trained(self):
        return self.epoch in self.train_accs
        
    def get_test_acc(self, F_test) -> float:
        acc = self.test_accs[F_test]
        return acc
    
    def get_test_loss(self, F_test) -> float:
        loss = self.test_losses[F_test]
        return loss
    
    def get_train_acc(self, epoch) -> np.ndarray:
        train_accs = self.train_accs[epoch]
        return train_accs
    
    def get_train_loss(self, epoch) -> np.ndarray:
        train_losses = self.train_losses[epoch]
        return train_losses
    
    def get_all_train_scores(self):
        return self.train_accs, self.train_losses
    
    def set_old_train_scores(self, train_accs, train_losses):
        self.train_accs.update(train_accs)
        self.train_losses.update(train_losses)
    
    def reset_test_scores(self):
        self.test_accs={}
        self.test_losses={}
    
    def add_train_scores(self, train_accs, train_losses):
        self.train_accs[self.epoch]=train_accs
        self.train_losses[self.epoch]=train_losses
        self.was_trained=True
        self.reset_test_scores()
        self.save()
    
    def add_test_scores(self, F_test, test_acc, test_loss):
        self.test_accs[F_test] = test_acc
        self.test_losses[F_test] = test_loss
        self.save()
        
    def check_train_scores(self):
        return self.epoch in self.train_accs and self.epoch in self.train_losses
        
                
                
class ModelUnit:
    build_params: BuildParams
    model_path: MyPath
    meta_file: MetaFile
    purge_old:bool
    verbose: bool
    previous_build_params: BuildParams
    previous_model_unit: "ModelUnit"
    monitors: list[Monitor]
    
    def __init__(
        self, 
        build_params: BuildParams,
        purge_old:bool=False ,
        monitors: list[Monitor]=[]
    ):
        '''
        Ensures the model is trained and tested when scores are requested
        '''
        # build parameters
        self.build_params=build_params
        
        self.monitors = monitors
        
        self.model_path = self.build_params.get_model_path()
        self.model_path.mkdir()
        
        self.meta_file=self.load_meta_file()
        
        # booleans
        self.purge_old=purge_old
        self.verbose=self.build_params.verbose
        
         # previous epoch
        self.previous_epoch_build_params=None
        self.previous_model_unit = None
    
        
        
    @staticmethod
    def load_units(
        build_params_array: np.ndarray[BuildParams],
        purge_old=False
    ): 
        model_units_array = np.vectorize(
            lambda build_params: ModelUnit(
                build_params=build_params, 
                purge_old=purge_old
            )
        )(build_params_array)
        return model_units_array
    
    def get_train_acc(self, epoch):
        assert 0 <= epoch <= self.build_params.epoch
        self.ensure_train_scores()
        
        return self.meta_file.get_train_acc(epoch)
    
    def get_all_train_scores(self):
        self.ensure_train_scores()
        
        return self.meta_file.get_all_train_scores()
    
    def get_train_loss(self, epoch):
        assert 0 <= epoch <= self.build_params.epoch
        self.ensure_train_scores()
        
        return self.meta_file.get_train_loss(epoch)
    
    def get_test_acc(self, F_test, epoch):
        if epoch > self.build_params.epoch:
            raise Exception('Invalid epoch value: Too high')
        
        if epoch == self.build_params.epoch:
            self.ensure_test_score(F_test)
            return self.meta_file.get_test_acc(F_test)
        
        self.load_previous_epoch_unit()
        return self.previous_model_unit.get_test_acc(F_test=F_test, epoch=epoch)

        
    
    def get_test_loss(self, F_test, epoch):
        if epoch > self.build_params.epoch:
            raise Exception('Invalid epoch value: Too high')
        
        if epoch == self.build_params.epoch:
            self.ensure_test_score(F_test)
            return self.meta_file.get_test_loss(F_test)
        
        self.load_previous_epoch_unit()
        return self.previous_model_unit.get_test_loss(F_test=F_test, epoch=epoch)

        
    def ensure_train_scores(self):
        # ensure meta_file exists
        if not self.meta_file.exists():
            self.meta_file.save()
            

        # ensure model is trained
        if self.purge_old or not self.check_trained():
            # ensure last epoch was trained
            if self.build_params.epoch > 0:
                self.load_previous_epoch_build_params()
                self.load_previous_epoch_unit()
            
            # train current epoch
            self.train()
        
    def ensure_test_score(self, F_test):
        self.ensure_train_scores()
        
        if not self.meta_file.check_F_test(F_test):
            self.test(F_test)
                
    def check_trained(self):
        model_file = self.build_params.get_model_file()
        return model_file.exists() and self.meta_file.check_train_scores()
    
    
    def load_meta_file(self):
        meta_file_path = self.build_params.get_meta_file()
        meta_file = meta_file_path.load()
        if meta_file is None:
            meta_file = MetaFile(meta_file_path, epoch=self.build_params.epoch)
        return meta_file
            
    def load_previous_epoch_build_params(self):
        self.previous_epoch_build_params = self.build_params.get_previous_epoch_params()
        
    def load_previous_epoch_unit(self):
        if self.previous_model_unit is None:
            previous_build_params = self.build_params.get_previous_epoch_params()
            self.previous_model_unit = ModelUnit(
                build_params=previous_build_params,
                purge_old=self.purge_old
            )
            self.meta_file.set_old_train_scores(*self.previous_model_unit.get_all_train_scores())
    
    def train(self, save_scores=True):
        train_losses, train_accs = ModelTrainer.train_unit(
            self.previous_epoch_build_params, 
            self.build_params
        )
        if save_scores:
            self.meta_file.add_train_scores(train_losses, train_accs)
        self.purge_old=False
        
    
    def test(self, F_test):
        test_acc, test_loss = ModelTrainer.test_unit(
                build_params=self.build_params,
                F_test = F_test
        )
        
        self.meta_file.add_test_scores(
            F_test=F_test,
            test_acc=test_acc,
            test_loss=test_loss
        )

    def pseudo_run(
        self, 
        experiment_params, 
        n_passes=1, 
        train_before_pseudo=False,
        pseudo_mode='test',
        pseudo_string=''
    ):
        if train_before_pseudo:
            self.ensure_train_scores()
        ModelTrainer.pseudo_run(
            build_params=self.build_params,
            experiment_params=experiment_params,
            monitors = self.monitors,
            n_passes=n_passes,
            use_trained_model=train_before_pseudo,
            pseudo_mode=pseudo_mode,
            pseudo_string=pseudo_string
        )