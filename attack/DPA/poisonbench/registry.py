"""Name -> class registries for transforms (attacks, benign edits), payloads and detectors.

To add an attack, write a file in ``transforms/`` and decorate its class with
``@register_transform``. To add a poisoned-data detector, do the same in
``detectors/`` with ``@register_detector``. Files in both folders are discovered
automatically.
"""

TRANSFORMS = {}
PAYLOADS = {}
DETECTORS = {}


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
register_detector = _registrar(DETECTORS, "detector")
get_transform_class = _getter(TRANSFORMS, "transform")
get_payload_class = _getter(PAYLOADS, "payload")
get_detector_class = _getter(DETECTORS, "detector")


def available_transforms():
    return sorted(TRANSFORMS)


def available_payloads():
    return sorted(PAYLOADS)


def available_detectors():
    return sorted(DETECTORS)
