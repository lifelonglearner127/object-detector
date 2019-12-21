import json

from json_minify import json_minify


class Conf:
    def __init__(self, conf_path):
        conf = json.loads(json_minify(open(conf_path).read()))
        self.__dict__.update(conf)

    def __getitem__(self, k):
        return self.__dict__.get(k, None)
