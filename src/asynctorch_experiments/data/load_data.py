
from asynctorch_experiments.buildtools.model_builder import BuildParams
from abc import ABC


class AsyncDataSet(ABC):
    def load_data(
            build_params:BuildParams, 
            train=False,
            test=False,
        ):
        pass

