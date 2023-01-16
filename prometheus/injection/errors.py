class MultipleInjectionError(Exception):
    """Raised when multiple types of injection events are found"""
    def __init__(self):
        self.message = "Multiple type of injection found. Can only do one at a time"
        super().__init__(self.message)

class TooManyInjectorsError(Exception):

    def __init__(self):
        self.message = "Multiple injectors found without specifying which to use. "
        self.message += "Please specify which key to use."
        super().__init__(self.message)

class DataNotLoadedError(Exception):
    """Raised if data isn't loaded"""

    def __init__(self):
        self.message = "You haven't loaded the data yet. Do `injection.load_data()` and try again"
        super().__init__(self.message)
