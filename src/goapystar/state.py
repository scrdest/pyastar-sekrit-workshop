import types
from copy import deepcopy


def statehash(goap_state: dict):
    # Used for transposition tables.
    # We want to skip equivalent plans, e.g. "Get[B] -> Get[A] -> Foo" == "Get[A] -> Get[B] -> Foo"
    # We don't care about the ordering if the results are equivalent,
    # and retaining such duplicates slows planning down significantly.
    # To do that, we need to have a way to check output states for any ordering.
    # We'll do that by just sorting keys alphabetically then stringifying them + value.
    if not isinstance(goap_state, dict):
        # for recursion
        return str(goap_state)

    sorted_keys = sorted(goap_state.keys(), key=lambda x: str(x))
    ordered_pairs = ((sortkey, goap_state[sortkey]) for sortkey in sorted_keys)
    recursivized = ((sortkey, statehash(sortval)) for (sortkey, sortval) in ordered_pairs)
    as_tuple = tuple(recursivized)
    hashed = hash(as_tuple)
    return hashed


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

    def as_hash(self):
        return statehash(self.to_dict())

    def __hash__(self):
        return statehash(self.to_dict())

    def __str__(self):
        stringform = f"<{self._name} ({self.to_dict()})>"
        return stringform
