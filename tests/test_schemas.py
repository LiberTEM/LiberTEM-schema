import pprint

import pytest

from pint import UnitRegistry
from pydantic_core import from_json

from libertem_schema import Simple4DSTEMParams, DimensionError

ureg = UnitRegistry()


def test_smoke():
    params = Simple4DSTEMParams(
        overfocus=0.0015 * ureg.meter,
        scan_pixel_pitch=0.000001 * ureg.meter,
        camera_length=0.15 * ureg.meter,
        detector_pixel_pitch=0.000050 * ureg.meter,
        semiconv=0.020 * ureg.radian,  # rad
        scan_rotation=330. * ureg.degree,
        flip_y=False,
        # Offset to avoid subchip gap in butted detectors
        cy=(32 - 2) * ureg.pixel,
        cx=(32 - 2) * ureg.pixel,
    )
    as_json = params.model_dump_json()
    print(as_json)
    from_j = from_json(as_json)
    assert Simple4DSTEMParams.model_validate(from_j)


def test_dimensionality():
    params = Simple4DSTEMParams(
        overfocus=0.0015 * ureg.degree,  # mismatch
        scan_pixel_pitch=0.000001 * ureg.meter,
        camera_length=0.15 * ureg.meter,
        detector_pixel_pitch=0.000050 * ureg.meter,
        semiconv=0.020 * ureg.radian,  # rad
        scan_rotation=330. * ureg.degree,
        flip_y=False,
        # Offset to avoid subchip gap in butted detectors
        cy=(32 - 2) * ureg.pixel,
        cx=(32 - 2) * ureg.pixel,
    )
    pprint.pprint(params)
    assert Simple4DSTEMParams.model_validate(params)
    as_json = params.model_dump_json()
    print(as_json)
    from_j = from_json(as_json)
    pprint.pprint(type(from_j['overfocus']))
    assert Simple4DSTEMParams.model_validate(from_j)
