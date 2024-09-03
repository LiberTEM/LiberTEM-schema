import pprint
import json

import jsonschema.exceptions
import pytest
import jsonschema

from pint import UnitRegistry, Quantity
from pydantic_core import from_json
from pydantic import ValidationError

from libertem_schema import Simple4DSTEMParams

ureg = UnitRegistry()


def test_smoke():
    params = Simple4DSTEMParams(
        overfocus=0.0015 * ureg.meter,
        scan_pixel_pitch=0.000001 * ureg.meter,
        camera_length=0.15 * ureg.meter,
        detector_pixel_pitch=0.000050 * ureg.meter,
        semiconv=0.020 * ureg.radian,
        scan_rotation=330. * ureg.degree,
        flip_y=False,
        cy=(32 - 2) * ureg.pixel,
        cx=(32 - 2) * ureg.pixel,
    )
    as_json = params.model_dump_json()
    pprint.pprint(("as json", as_json))
    from_j = from_json(as_json)
    pprint.pprint(("from json", from_j))
    res = Simple4DSTEMParams.model_validate(from_j)
    pprint.pprint(("validated", res))
    assert isinstance(res.overfocus, Quantity)
    assert isinstance(res.flip_y, bool)
    assert res == params


def test_angstrom():
    params = Simple4DSTEMParams(
        overfocus=0.0015 * ureg.meter,
        scan_pixel_pitch=10000 * ureg.angstrom,
        camera_length=0.15 * ureg.meter,
        detector_pixel_pitch=0.000050 * ureg.meter,
        semiconv=0.020 * ureg.radian,
        scan_rotation=330. * ureg.degree,
        flip_y=False,
        cy=(32 - 2) * ureg.pixel,
        cx=(32 - 2) * ureg.pixel,
    )
    as_json = params.model_dump_json()
    pprint.pprint(("as json", as_json))
    from_j = from_json(as_json)
    pprint.pprint(("from json", from_j))
    res = Simple4DSTEMParams.model_validate(from_j)
    pprint.pprint(("validated", res))
    assert isinstance(res.overfocus, Quantity)
    # To JSON converts to SI base units
    assert res.scan_pixel_pitch.units == 'meter'
    assert isinstance(res.flip_y, bool)
    assert res == params


def test_missing():
    with pytest.raises(ValidationError):
        Simple4DSTEMParams(
            # Missing!
            # overfocus=0.0015 * ureg.meter,
            scan_pixel_pitch=0.000001 * ureg.meter,
            camera_length=0.15 * ureg.meter,
            detector_pixel_pitch=0.000050 * ureg.meter,
            semiconv=0.020 * ureg.radian,
            scan_rotation=330. * ureg.degree,
            flip_y=False,
            cy=(32 - 2) * ureg.pixel,
            cx=(32 - 2) * ureg.pixel,
        )


def test_carrots():
    with pytest.raises(ValidationError):
        Simple4DSTEMParams(
            overfocus=0.0015,  # carrots
            scan_pixel_pitch=0.000001 * ureg.meter,
            camera_length=0.15 * ureg.meter,
            detector_pixel_pitch=0.000050 * ureg.meter,
            semiconv=0.020 * ureg.radian,
            scan_rotation=330. * ureg.degree,
            flip_y=False,
            cy=(32 - 2) * ureg.pixel,
            cx=(32 - 2) * ureg.pixel,
        )


def test_dimensionality():
    with pytest.raises(ValidationError):
        Simple4DSTEMParams(
            # dimensionality mismatch!
            overfocus=0.0015 * ureg.degree,
            ###
            scan_pixel_pitch=0.000001 * ureg.meter,
            camera_length=0.15 * ureg.meter,
            detector_pixel_pitch=0.000050 * ureg.meter,
            semiconv=0.020 * ureg.radian,
            scan_rotation=330. * ureg.degree,
            flip_y=False,
            cy=(32 - 2) * ureg.pixel,
            cx=(32 - 2) * ureg.pixel,
        )


def test_json_dimension():
    params = Simple4DSTEMParams(
        overfocus=0.0015 * ureg.meter,
        scan_pixel_pitch=0.000001 * ureg.meter,
        camera_length=0.15 * ureg.meter,
        detector_pixel_pitch=0.000050 * ureg.meter,
        semiconv=0.020 * ureg.radian,
        scan_rotation=330. * ureg.degree,
        flip_y=False,
        cy=(32 - 2) * ureg.pixel,
        cx=(32 - 2) * ureg.pixel,
    )
    as_json = params.model_dump_json()
    from_j = from_json(as_json)
    # Mess up dimensionality
    from_j['overfocus'][1] = 'degree'
    with pytest.raises(ValidationError):
        Simple4DSTEMParams.model_validate(from_j)


def test_json_repr():
    params = Simple4DSTEMParams(
        overfocus=0.0015 * ureg.meter,
        scan_pixel_pitch=0.000001 * ureg.meter,
        camera_length=0.15 * ureg.meter,
        detector_pixel_pitch=0.000050 * ureg.meter,
        semiconv=0.020 * ureg.radian,
        scan_rotation=330. * ureg.degree,
        flip_y=False,
        cy=(32 - 2) * ureg.pixel,
        cx=(32 - 2) * ureg.pixel,
    )
    as_json = params.model_dump_json()
    from_j = from_json(as_json)
    # Mess up plain type representation of Quantity as (float, str)
    from_j['overfocus'].append('hurz')
    with pytest.raises(ValidationError):
        Simple4DSTEMParams.model_validate(from_j)


def test_json_missing():
    params = Simple4DSTEMParams(
        overfocus=0.0015 * ureg.meter,
        scan_pixel_pitch=0.000001 * ureg.meter,
        camera_length=0.15 * ureg.meter,
        detector_pixel_pitch=0.000050 * ureg.meter,
        semiconv=0.020 * ureg.radian,
        scan_rotation=330. * ureg.degree,
        flip_y=False,
        # Offset to avoid subchip gap in butted detectors
        cy=(32 - 2) * ureg.pixel,
        cx=(32 - 2) * ureg.pixel,
    )
    as_json = params.model_dump_json()
    from_j = from_json(as_json)
    # Missing key
    del from_j['overfocus']
    with pytest.raises(ValidationError):
        Simple4DSTEMParams.model_validate(from_j)


def test_json_schema_smoke():
    params = Simple4DSTEMParams(
        overfocus=1.5 * ureg.millimeter,
        scan_pixel_pitch=0.000001 * ureg.meter,
        camera_length=0.15 * ureg.meter,
        detector_pixel_pitch=0.000050 * ureg.meter,
        semiconv=0.020 * ureg.radian,
        scan_rotation=330. * ureg.degree,
        flip_y=False,
        # Offset to avoid subchip gap in butted detectors
        cy=(32 - 2) * ureg.pixel,
        cx=(32 - 2) * ureg.pixel,
    )
    json_schema = params.model_json_schema()
    pprint.pprint(json_schema)
    as_json = params.model_dump_json()
    pprint.pprint(as_json)
    loaded = json.loads(as_json)
    jsonschema.validate(
        instance=loaded,
        schema=json_schema
    )
    # JSON is in SI base units
    assert tuple(loaded['overfocus']) == (0.0015, 'meter')


def test_json_schema_repr():
    params = Simple4DSTEMParams(
        overfocus=0.0015 * ureg.meter,
        scan_pixel_pitch=0.000001 * ureg.meter,
        camera_length=0.15 * ureg.meter,
        detector_pixel_pitch=0.000050 * ureg.meter,
        semiconv=0.020 * ureg.radian,
        scan_rotation=330. * ureg.degree,
        flip_y=False,
        # Offset to avoid subchip gap in butted detectors
        cy=(32 - 2) * ureg.pixel,
        cx=(32 - 2) * ureg.pixel,
    )
    json_schema = params.model_json_schema()
    pprint.pprint(json_schema)
    as_json = params.model_dump_json()
    pprint.pprint(as_json)
    loaded = json.loads(as_json)
    loaded['overfocus'].append('3')
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(
            instance=loaded,
            schema=json_schema
        )


def test_json_schema_missing():
    params = Simple4DSTEMParams(
        overfocus=0.0015 * ureg.meter,
        scan_pixel_pitch=0.000001 * ureg.meter,
        camera_length=0.15 * ureg.meter,
        detector_pixel_pitch=0.000050 * ureg.meter,
        semiconv=0.020 * ureg.radian,
        scan_rotation=330. * ureg.degree,
        flip_y=False,
        # Offset to avoid subchip gap in butted detectors
        cy=(32 - 2) * ureg.pixel,
        cx=(32 - 2) * ureg.pixel,
    )
    json_schema = params.model_json_schema()
    pprint.pprint(json_schema)
    as_json = params.model_dump_json()
    pprint.pprint(as_json)
    loaded = json.loads(as_json)
    del loaded['overfocus']
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(
            instance=loaded,
            schema=json_schema
        )


# No dimensionality check in JSON Schema yet
@pytest.mark.xfail
def test_json_schema_dim():
    params = Simple4DSTEMParams(
        overfocus=0.0015 * ureg.meter,
        scan_pixel_pitch=0.000001 * ureg.meter,
        camera_length=0.15 * ureg.meter,
        detector_pixel_pitch=0.000050 * ureg.meter,
        semiconv=0.020 * ureg.radian,
        scan_rotation=330. * ureg.degree,
        flip_y=False,
        # Offset to avoid subchip gap in butted detectors
        cy=(32 - 2) * ureg.pixel,
        cx=(32 - 2) * ureg.pixel,
    )
    json_schema = params.model_json_schema()
    pprint.pprint(json_schema)
    as_json = params.model_dump_json()
    pprint.pprint(as_json)
    loaded = json.loads(as_json)
    loaded['overfocus'][1] = 'degree'
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(
            instance=loaded,
            schema=json_schema
        )
