import json
import os
import typing

from goapystar.reasoning.utils import State


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = CURR_DIR

STATE_MARKER = "_isState"
NAME_MARKER = "_stateName"


class MapJsonEncoder(json.JSONEncoder):
    def default(self, o: typing.Any) -> typing.Any:
        if isinstance(o, State):
            raw_serialized = o.to_dict()
            raw_serialized[STATE_MARKER] = True
            raw_serialized[NAME_MARKER] = o._name
            return raw_serialized

        return super().default(o)


class MapJsonDecoder(json.JSONDecoder):
    def decode(self, o: str, *args, **kwargs) -> typing.Any:
        pre_decoded = super().decode(o, *args, **kwargs)
        result = {
            raw_key: [
                State.fromdict(item, name=item.get(NAME_MARKER)) if isinstance(item, dict) and item.get(STATE_MARKER)
                else item
                for item in raw_row
            ]
            for raw_key, raw_row
            in pre_decoded.items()
        }
        return result


def save_map_pickle(mapdata, filename="map"):
    import pickle
    savepath = os.path.join(MAPS_DIR, f"{filename}.pkl")

    with open(savepath, "wb") as mapfile:
        pickle.dump(mapdata, mapfile)

    return savepath


def save_map_json(mapdata, filename="map"):
    savepath = os.path.join(MAPS_DIR, f"{filename}.json")

    with open(savepath, "w") as mapfile:
        json.dump(mapdata, mapfile, indent=4, cls=MapJsonEncoder)

    return savepath


def load_map_json(filename="map"):
    loadpath = os.path.join(MAPS_DIR, f"{filename}.json")

    with open(loadpath, "r") as mapfile:
        data = json.load(mapfile, cls=MapJsonDecoder)

    return data
