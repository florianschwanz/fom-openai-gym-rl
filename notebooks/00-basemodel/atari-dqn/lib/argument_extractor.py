class ArgumentExtractor:

    def extract_argument(kwargs, name, default_value):
        return kwargs[name] if name in kwargs else default_value
