# abstract base class that handles a model
# build on top of https://github.com/lightly-ai/lightly-train?tab=readme-ov-file

class vModel(abstract):
    innit():
        self.data_path
        self.model

    @abstract
    def train()

    @abstract
    def load()

    @abstract
    def predict()

    @abstract
    def eval()
