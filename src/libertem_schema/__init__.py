from typing import Any, Sequence, Callable, Union
import functools

from typing_extensions import TypeVar, Generic, get_args, get_origin, Annotated

import numpydantic
import numpy as np

from pydantic_core import core_schema
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    ValidationInfo,
    TypeAdapter
)
from pydantic.errors import PydanticSchemaGenerationError

import pint


__version__ = '0.1.0.dev0'

ureg = pint.UnitRegistry()


class DimensionError(ValueError):
    pass



def to_tuple(q: pint.Quantity, base_units: pint.Unit):
    base = q.to(base_units)
    return (base.magnitude, str(base.units))


def to_array_tuple(q: pint.Quantity, info: ValidationInfo, base_units: pint.Unit, array_serializer: Callable):
    base = q.to(base_units)
    return (array_serializer(base.magnitude, info=info), str(base.units))


def get_basic_type(t):
    if isinstance(t, Sequence) and not isinstance(t, str):
        # numpydantic.dtype.Float is a sequence, for example They all map to the
        # same basic Python type

        # FIXME for some reason they don't work:
        # TypeError: Too many parameters for <class
        # 'libertem_schema._make_type.<locals>.Single'>; actual 5, expected 1
        t = t[0]
    origin = get_origin(t)
    if origin is not None and issubclass(origin, Annotated):
        args = get_args(t)
        # First argument is the bas type
        t = args[0]

    if t in (float, int, complex):
        return t
    dtype = np.dtype(t)
    return numpydantic.maps.np_to_python[dtype.type]


def get_schema(t):
    basic_type = get_basic_type(t)
    if basic_type is float:
        return core_schema.float_schema()
    elif basic_type is int:
        return core_schema.int_schema()
    elif basic_type is complex:
        return core_schema.complex_schema()
    else:
        raise NotImplementedError(t)


def make_type(reference: pint.Quantity):

    DType = TypeVar('DType')
    Shape = TypeVar('Shape')

    class Single(pint.Quantity, Generic[DType]):
        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source_type: Any,
            _handler: GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
            (dtype, ) = _source_type.__args__
            magnitude_schema = get_schema(dtype)
            units = str(reference.units)
            try:
                adapter = TypeAdapter(dtype)
                validate_function = adapter.validate_python
                magnitude_schema = adapter.core_schema
            except PydanticSchemaGenerationError as e:
                validate_function = numpydantic.schema.get_validate_interface(Any, dtype)
                magnitude_schema = get_schema(dtype)
            target_type = get_basic_type(dtype)

            def validator(v: Any, info: ValidationInfo) -> pint.Quantity:
                if isinstance(v, pint.Quantity):
                    pass
                elif isinstance(v, Sequence):
                    magnitude, unit = v
                    v = pint.Quantity(value=magnitude, units=unit)
                else:
                    raise ValueError(f"Don't know how to interpret type {type(v)}.")
                # Check dimension
                if not v.check(reference.dimensionality):
                    raise DimensionError(f"Expected dimensionality {reference.dimensionality}, got quantity {v}.")
                try:
                    # First, try as-is
                    validate_function(v.magnitude)
                except Exception:
                    # See if we can go from int to float, for example
                    if np.can_cast(type(v.magnitude), target_type):
                        v = target_type(v.magnitude) * v.units
                        validate_function(v.magnitude)
                    else:
                        raise
                # Return target type
                return v

            json_schema = core_schema.tuple_positional_schema(items_schema=[
                magnitude_schema,
                core_schema.literal_schema([units])
            ])
            return core_schema.json_or_python_schema(
                json_schema=json_schema,
                python_schema=core_schema.with_info_plain_validator_function(validator),
                serialization=core_schema.plain_serializer_function_ser_schema(
                    functools.partial(to_tuple, base_units=reference.units)
                ),
            )

    class Array(pint.Quantity, Generic[Shape, DType]):
        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source_type: Any,
            _handler: GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
            shape, dtype = _source_type.__args__
            numpydantic_type = numpydantic.NDArray[shape, dtype]
            units = str(reference.to_base_units().units)
            validate_function = numpydantic.schema.get_validate_interface(shape, dtype)
            target_type = get_basic_type(dtype)

            magnitude_schema = numpydantic_type.__get_pydantic_core_schema__(
                _source_type=numpydantic_type,
                _handler=_handler
            )

            def validator(v: Any, info: core_schema.ValidationInfo) -> pint.Quantity:
                if isinstance(v, pint.Quantity):
                    pass
                elif isinstance(v, Sequence):
                    magnitude, unit = v
                    # Turn into Quantity: magnitude * unit
                    v = pint.Quantity(value=np.asarray(magnitude), units=unit)
                else:
                    raise ValueError(f"Don't know how to interpret type {type(v)}.")
                # Check dimension
                if not v.check(reference.dimensionality):
                    raise DimensionError(f"Expected dimensionality {reference.dimensionality}, got quantity {v}.")
                try:
                    # First, try as-is
                    validate_function(v.magnitude)
                except Exception:
                    # See if we can go from int to float, for example
                    if np.can_cast(v.magnitude, target_type):
                        v = v.magnitude.astype(target_type) * v.units
                        validate_function(v.magnitude)
                    else:
                        raise
                # Return target type
                return v

            json_schema = core_schema.tuple_positional_schema(items_schema=[
                magnitude_schema['json_schema'],
                core_schema.literal_schema([units])
            ])

            serializer = functools.partial(
                to_array_tuple,
                base_units=reference.units,
                array_serializer=magnitude_schema['serialization']['function']
            )
            return core_schema.json_or_python_schema(
                json_schema=json_schema,
                python_schema=core_schema.with_info_plain_validator_function(validator),
                serialization=core_schema.plain_serializer_function_ser_schema(
                    function=serializer,
                    info_arg=True,
                ),
            )

    return Single, Array


Length, LengthArray = make_type(pint.Quantity(1, 'meter'))
Angle, AngleArray = make_type(pint.Quantity(1, 'radian'))
Pixel, PixelArray = make_type(pint.Quantity(1, 'pixel'))


class Simple4DSTEMParams(BaseModel):
    '''
    Basic calibration parameters of a strongly simplified model
    of a 4D STEM experiment.

    See https://github.com/LiberTEM/Microscope-Calibration
    and https://arxiv.org/abs/2403.08538
    for the technical details.
    '''
    overfocus: Length[float]
    scan_pixel_pitch: Length[float]
    camera_length: Length[float]
    detector_pixel_pitch: Length[float]
    semiconv: Angle[float]
    cy: Pixel[float]
    cx: Pixel[float]
    scan_rotation: Angle[float]
    flip_y: bool
