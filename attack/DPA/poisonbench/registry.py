"""Name -> class registries for transforms (attacks, benign edits) and payloads.

To add an attack, write a file in ``transforms/`` and decorate its class with
``@register_transform``. Files in that folder are discovered automatically.
"""

TRANSFORMS = {}
PAYLOADS = {}


def _registrar(table, kind):
    def register(cls):
        if cls.name in table:
            raise ValueError(f"{kind} {cls.name!r} registered twice")
        table[cls.name] = cls
        return cls
    return register


def _getter(table, kind):
    def get(name):
        if name not in table:
            raise KeyError(f"unknown {kind} {name!r}; available: {', '.join(sorted(table))}")
        return table[name]
    return get


register_transform = _registrar(TRANSFORMS, "transform")
register_payload = _registrar(PAYLOADS, "payload")
get_transform_class = _getter(TRANSFORMS, "transform")
get_payload_class = _getter(PAYLOADS, "payload")


def available_transforms():
    return sorted(TRANSFORMS)


def available_payloads():
    return sorted(PAYLOADS)
