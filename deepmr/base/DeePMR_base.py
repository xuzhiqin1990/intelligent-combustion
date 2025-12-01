import os


class DeePMR_base_class():
    def __init__(
            self,
            settings,
    ):
        self.working_dir = settings.working_dir
        self.species_num = settings.species_num
        self.indicators = settings.indicators
        self.dnn_assist = settings.DNN_assist
        self.retained_index = []

        self.generate_dir()
        self.load_path()

    def set_retained_index(self, retained_index):
        self.retained_index = list(retained_index)


    def generate_dir(self):
        r"""
        生成保存数据的文件夹
        """
        for i, indicator in enumerate(self.indicators):
            if self.dnn_assist[i] == True:
                os.makedirs(
                    f'{self.working_dir}/data/dnn_data/{indicator}', exist_ok=True)
                os.makedirs(
                    f'{self.working_dir}/models/model_{indicator}/model_pth', exist_ok=True)
                os.makedirs(
                    f'{self.working_dir}/models/model_{indicator}/loss_his', exist_ok=True)
                os.makedirs(
                    f'{self.working_dir}/models/model_{indicator}/config_json', exist_ok=True)
                os.makedirs(
                    f'{self.working_dir}/models/model_{indicator}/training_log', exist_ok=True)

        os.makedirs(f'{self.working_dir}/data/dnn_data/input', exist_ok=True)
        os.makedirs(f'{self.working_dir}/data/vector_data', exist_ok=True)
        os.makedirs(f'{self.working_dir}/data/simulation_data', exist_ok=True)
        os.makedirs(
            f'{self.working_dir}/data/father_sample_data', exist_ok=True)
        os.makedirs(f'{self.working_dir}/log/gendata', exist_ok=True)
        os.makedirs(f'{self.working_dir}/log/screening', exist_ok=True)

    def load_path(self):
        r"""
        加载保存数据的文件夹路径
        """
        self.vector_data_path = f'{self.working_dir}/data/vector_data'
        self.father_sample_data_path = f'{self.working_dir}/data/father_sample_data'
        self.log_path = f'{self.working_dir}/log'
        self.log_gen_data_path = f'{self.working_dir}/log/gendata'
        self.log_screening_path = f'{self.working_dir}/log/screening'
        self.simulation_data_path = f'{self.working_dir}/data/simulation_data'
        self.dnn_input_data_path  = f'{self.working_dir}/data/dnn_data/input'

        self.data_path = {}
        self.dnn_data_path = {}
        self.model_path = {}
        self.model_pth_path = {}
        self.loss_his_path = {}
        self.train_log_his_path = {}
        self.model_json_path = {}

        for indicator in self.indicators:
            self.data_path[indicator] = f'{self.working_dir}/data/{indicator}'
            self.dnn_data_path[indicator] = f'{self.working_dir}/data/dnn_data/{indicator}'
            self.model_path[indicator] = f'{self.working_dir}/models/model_{indicator}'
            self.model_pth_path[indicator] = f'{self.working_dir}/models/model_{indicator}/model_pth'
            self.loss_his_path[indicator] = f'{self.working_dir}/models/model_{indicator}/loss_his'
            self.train_log_his_path[indicator] = f'{self.working_dir}/models/model_{indicator}/training_log'
            self.model_json_path[indicator] = f'{self.working_dir}/models/model_{indicator}/config_json'
