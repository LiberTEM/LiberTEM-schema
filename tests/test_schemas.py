import pprint
import json

import jsonschema.exceptions
import pytest
import jsonschema
import numpy as np

from pint import UnitRegistry, Quantity
from pydantic_core import from_json
from pydantic import ValidationError, BaseModel

from numpydantic import Shape

from libertem_schema import Simple4DSTEMParams, Length, LengthArray

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


def test_nocast():
    class T1(BaseModel):
        t: Length[int]

    with pytest.raises(ValidationError):
        t1 = T1(t=Quantity(0.3, 'm'))

    class T2(BaseModel):
        t: LengthArray[Shape['2 x, 2 y'], int]

    with pytest.raises(ValidationError):
        t2 = T2(t=Quantity(
            np.array([(1, 2), (3, 4)]).astype(float),
            'm'
        ))


def test_json_nocast():
    class T1(BaseModel):
        t: Length[int]

    params = T1(t=Quantity(1, 'm'))
    
    json_schema = params.model_json_schema()
    pprint.pprint(json_schema)
    as_json = params.model_dump_json()
    pprint.pprint(as_json)
    loaded = json.loads(as_json)
    loaded['t'][0] = 0.3

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(
            instance=loaded,
            schema=json_schema
        )

    class T2(BaseModel):
        t: LengthArray[Shape['2 x, 2 y'], int]

    params = T2(t=Quantity(
        np.array([(1, 2), (3, 4)]),
        'm'
    ))

    json_schema = params.model_json_schema()
    pprint.pprint(json_schema)
    as_json = params.model_dump_json()
    pprint.pprint(as_json)
    loaded = json.loads(as_json)
    loaded['t'][0] = [[0.3, 0.4], [0.5, 0.6]]

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(
            instance=loaded,
            schema=json_schema
        )


def test_shape():
    class T(BaseModel):
        t: LengthArray[Shape['2 x, 2 y'], complex]

    with pytest.raises(ValidationError):
        t = T(t=Quantity(
            # Shape mismatch
            np.array([(1, 2), (3, 4), (5, 6)]).astype(float),
            'm'
        ))


def test_json_shape():
    class T2(BaseModel):
        t: LengthArray[Shape['2 x, 2 y'], int]

    params = T2(t=Quantity(
        np.array([(1, 2), (3, 4)]),
        'm'
    ))

    json_schema = params.model_json_schema()
    pprint.pprint(json_schema)
    as_json = params.model_dump_json()
    pprint.pprint(as_json)
    loaded = json.loads(as_json)
    loaded['t'][0] = [[1, 2], [3, 4], [5, 6]]

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(
            instance=loaded,
            schema=json_schema
        )


def test_dtypes():
    class T(BaseModel):
        t: LengthArray[Shape['2 x, 2 y'], np.complex128]

    t = T(t=Quantity(
        # Shape mismatch
        np.array([(1, 2), (3, 4)]),
        'm'
    ))


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
