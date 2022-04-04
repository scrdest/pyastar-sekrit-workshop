import types
from copy import deepcopy


class State(types.SimpleNamespace):
    def __init__(self, name=None, **values):
        super().__init__()
        self.__dict__.update(values)
        self._name = name or "State"

    @classmethod
    def fromdict(cls, initdict: dict, name=None):
        inst = cls()
        inst.__dict__.update(initdict)
        if name:
            inst._name = name
        return inst

    def to_dict(self):
        result = dict(self.items())
        return result

    def keys(self):
        return (k for k in self.__dict__.keys() if not k[0][0] == "_")

    def items(self):
        return (i for i in self.__dict__.items() if not i[0][0] == "_")

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __deepcopy__(self, memodict):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v))
        return result


    def __getattr__(self, item):
        if item in self.__dict__:
            result = self.__dict__[item]
        else:
            result = super().__getattribute__(item)
        return result

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return other.to_dict() == self.to_dict()

        return super().__eq__(other)

    def __hash__(self):
        return id(self)

    def __str__(self):
        stringform = f"<{self._name} ({self.to_dict()})>"
        return stringform
