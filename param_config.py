class ParamConfig:
    def __init__(self,
                 data_path,
                 processed_data_path,
                 stemmer_type):
        self.data_path = data_path
        self.processed_data_path = processed_data_path
        self.stemmer_type = stemmer_type

config = ParamConfig(data_path = "./data",
                     processed_data_path = "./processed_data",
                     stemmer_type = "snowball")

