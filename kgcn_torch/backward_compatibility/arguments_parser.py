class ArgsHandler:
    """ Handling Arguments class
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def dump_json(self, filename):
        """ dump arguments into json file
        """
        return [filename, self.args, self.kwargs]

    def dump_yml(self, filename):
        """ dump arguments into yaml file
        """
        return [filename, self.args, self.kwargs]

    @classmethod
    def read_json(cls, filename) -> "ArgsHandler":
        """ read configure from json file
        """
        return [filename]


    @classmethod
    def read_yml(cls, filename) -> "ArgsHandler":
        """ read configure from yaml file
        """
        return [filename]
