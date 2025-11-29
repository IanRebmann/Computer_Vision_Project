from Computer_Vision_Project.carlos.colorization_object import ColorizationModel

class Models:
    def __init__(
            self,
    ):
        self.colorization_model = ColorizationModel()
        #self.state_dict = load_file(self.weights_init)
        #self.l_model = model_init(self.state_dict)

initialized_models = Models()
