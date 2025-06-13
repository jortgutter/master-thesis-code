import os
import torch
import pickle
import json
import csv
import gc  
import matplotlib.pyplot as plt

class MyPath(str):

    def exists(self):
        return os.path.exists(self)
    
    def mkdir(self, recursive=False):
        if not self.exists():
            parent_path = self.change_dir('..')
            if recursive or parent_path.exists():
                os.makedirs(self)
            else:
                raise Exception(f'Parent path does not exist ({parent_path})')

    def change_dir(self, *relative_path):
        # split relative path into steps
        def split_sub_path(relative_path):
            return [
                step
                for relative_sub_path in relative_path
                for step in relative_sub_path.split(os.path.sep) 
                if step
            ]
        

        relative_path_steps = split_sub_path([*relative_path])
        
        # start at self
        cwd = self
        
        # follow path
        for step in relative_path_steps:
            if step == '..':
                # go up a level
                cwd = os.path.dirname(cwd)
            elif step == '.':
                # remain in cwd
                pass
            else:
                # go down a level
                cwd = os.path.join(cwd, step)
                
        # create new path instance of destination
        return MyPath(cwd)
        
    def get_folder_name(self):
        return self.split(os.path.sep)[-1]
    
    def file(self, file_name, extension='txt'):
        return MyFile(
            file_name=file_name,
            extension=extension,
            folder_path=self
        )
        
    
class MyFile:
    def __init__(
        self, 
        file_name:str, 
        extension:str, 
        folder_path:MyPath
    ):
        self.file_name=file_name
        self.extension=extension
        self.folder_path=folder_path
        self.proper_file_name = f'{file_name}.{extension}'
        self.file_path = os.path.join(self.folder_path, self.proper_file_name)

    def exists(self):
        return os.path.exists(self.file_path)
    
    def folder_exists(self):
        return self.folder_path.exists()
    
    def create_folder(self, recursive=False):
        self.folder_path.mkdir(recursive=recursive)
        
    def save(self, file_contents, force_create_path=False):
        if force_create_path:
            self.create_folder(recursive=True)
            
        if self.extension == 'pth':
            torch.save(file_contents, self.file_path)
            
        elif self.extension == 'pkl':
            with open(self.file_path, "wb") as f:
                pickle.dump(file_contents, f)
                
        elif self.extension == 'png':
            plt.savefig(self.file_path)
                
        elif self.extension == 'json':
            assert type(file_contents) in [dict, list[dict]]
            with open(self.file_path, "w") as f:
                json.dump(file_contents, f, indent=4)
                
        else:
            raise NotImplementedError(f'Filetype {self.extension} not supported (yet)')
                    
        gc.collect()
        
        
    def load(self):
        data = None
        if not self.exists():
            return data
        
        if self.extension == 'pth':
            data = torch.load(self.file_path)
            
        elif self.extension == 'pkl':
            with open(self.file_path, "rb") as f:
                data = pickle.load(f)
                
        elif self.extension == 'json':
            with open(self.file_path, "r") as f:
                data = json.load(f)
                
        else:
            raise NotImplementedError(f'Filetype {self.extension} not supported (yet)')
                    
        gc.collect()
        return data
