import json
import os
import typing

from goapystar.state import State
from goapystar.constants import MAPS_DIR, STATE_MARKER, NAME_MARKER


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
            if not raw_key[0] == "_"
        }
        return result


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