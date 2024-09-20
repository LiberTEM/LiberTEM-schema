import pprint
import json

import jsonschema.exceptions
import pytest
import jsonschema
import numpy as np

from typing_extensions import Annotated, get_origin, Generic
from pint import UnitRegistry, Quantity
from pydantic_core import from_json
from pydantic import ValidationError, BaseModel, PositiveFloat, Field

from numpydantic import Shape
import numpydantic.dtype

from libertem_schema import (
    Simple4DSTEMParams, Length, LengthArray, Single, Array, DType as DType_TVar, Shape as Shape_TVar
)

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
        T1(t=Quantity(0.3, 'm'))

    class T2(BaseModel):
        t: LengthArray[Shape['2 x, 2 y'], int]

    with pytest.raises(ValidationError):
        T2(t=Quantity(
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
        T(t=Quantity(
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


def test_dtype():
    class T(BaseModel):
        t: Length[np.complex128]

    T(t=Quantity(23, 'm'))


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


def test_json_schema_array():
    class T(BaseModel):
        t: LengthArray[Shape['2 x, 2 y'], float]

    params = T(t=Quantity(
        np.array([
            (1, 2),
            (3, 4)
        ]),
        'cm'
    ))
    json_schema = params.model_json_schema()
    pprint.pprint(json_schema)
    as_json = params.model_dump_json()
    pprint.pprint(as_json)
    loaded = json.loads(as_json)
    jsonschema.validate(
        instance=loaded,
        schema=json_schema
    )


@pytest.mark.xfail
def test_json_schema_complex_array():
    '''
    No native support for complex numbers in JSON
    '''
    class T(BaseModel):
        t: LengthArray[Shape['2 x, 2 y'], complex]

    params = T(t=Quantity(
        np.array([
            (1, 2),
            (3, 4)
        ]),
        'cm'
    ))
    json_schema = params.model_json_schema()
    pprint.pprint(json_schema)
    as_json = params.model_dump_json()
    pprint.pprint(as_json)
    loaded = json.loads(as_json)
    jsonschema.validate(
        instance=loaded,
        schema=json_schema
    )


@pytest.mark.xfail
def test_json_schema_complex():
    '''
    No native support for complex numbers in JSON
    '''
    class T(BaseModel):
        t: Length[complex]

    params = T(t=Quantity(1, 'cm'))
    json_schema = params.model_json_schema()
    pprint.pprint(json_schema)
    as_json = params.model_dump_json()
    pprint.pprint(as_json)
    loaded = json.loads(as_json)
    jsonschema.validate(
        instance=loaded,
        schema=json_schema
    )


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


@pytest.mark.parametrize(
    "dtype", (
        float,
        numpydantic.dtype.Float,
        Annotated[float, Field(strict=False, gt=0)],
        np.float64,
        PositiveFloat
    )
)
@pytest.mark.parametrize(
    "array", (True, False)
)
def test_dtypes(dtype, array):
    if dtype is numpydantic.dtype.Float:
        pytest.xfail(
            "FIXME find out how the numpydantic generic types can be integrated, "
            "can't use them as argument as of now"
        )
    if array:
        origin = get_origin(dtype)
        if origin is not None and issubclass(origin, Annotated):
            pytest.xfail("FIXME make arrays and pydantic types compatible, somehow.")

        class T(BaseModel):
            t: LengthArray[Shape['2 x, 2 y'], dtype]

        t = T(t=Quantity(
            np.array([(1., 2.), (3., 4.)]),
            'm'
        ))
    else:
        class T(BaseModel):
            t: Length[dtype]

        t = T(t=Quantity(0.3, 'm'))

    json_schema = t.model_json_schema()
    pprint.pprint(json_schema)
    as_json = t.model_dump_json()
    pprint.pprint(as_json)
    t.model_validate_json(as_json)
    loaded = json.loads(as_json)
    t.model_validate(loaded)
    jsonschema.validate(
        instance=loaded,
        schema=json_schema
    )


# we set the base unit to cm
_cm = Quantity(1, 'cm')


class Cm(Single, Generic[DType_TVar]):
    reference = _cm


class CmArray(Array, Generic[Shape_TVar, DType_TVar]):
    reference = _cm


def test_other_unit():
    class T1(BaseModel):
        t: Cm[float]

    class T2(BaseModel):
        t: CmArray[Shape['2 x, 2 y'], float]

    t1 = T1(t=Quantity(0.3, 'm'))
    json_schema = t1.model_json_schema()
    pprint.pprint(json_schema)
    as_json = t1.model_dump_json()
    pprint.pprint(as_json)
    t1.model_validate_json(as_json)
    loaded = json.loads(as_json)
    assert loaded['t'][0] == 30
    assert loaded['t'][1] == 'centimeter'
    t1.model_validate(loaded)
    jsonschema.validate(
        instance=loaded,
        schema=json_schema
    )

    t2 = T2(t=Quantity([(0.3, 0.3), (0.3, 0.3)], 'm'))
    json_schema = t2.model_json_schema()
    pprint.pprint(json_schema)
    as_json = t2.model_dump_json()
    pprint.pprint(as_json)
    t2.model_validate_json(as_json)
    loaded = json.loads(as_json)
    assert loaded['t'][0] == [[30, 30], [30, 30]]
    assert loaded['t'][1] == 'centimeter'
    t2.model_validate(loaded)
    jsonschema.validate(
        instance=loaded,
        schema=json_schema
    )
