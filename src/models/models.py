

# model template
class Model:
    def __init__(self, args, **kwargs):
        pass

    def predict(self, data) -> str:
        pass

    def post_process(self, raw_text_output) -> str:
        pass
