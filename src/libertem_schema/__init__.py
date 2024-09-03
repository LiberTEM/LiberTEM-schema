from typing import Any, Tuple, Sequence
from numbers import Number

from typing_extensions import Annotated
from pydantic_core import core_schema
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    WrapValidator,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
)

from pydantic.json_schema import JsonSchemaValue
import pint


__version__ = '0.1.0.dev0'

ureg = pint.UnitRegistry()


class DimensionError(ValueError):
    pass


_pint_base_repr = core_schema.tuple_positional_schema(items_schema=[
    core_schema.float_schema(),
    core_schema.str_schema()
])


class PintAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate_from_tuple(value: Tuple[Number, str]) -> pint.Quantity:
            m, u = value
            return m * ureg(u)

        from_tuple_schema = core_schema.chain_schema(
            [
                _pint_base_repr,
                core_schema.no_info_plain_validator_function(validate_from_tuple),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_tuple_schema,
            python_schema=core_schema.chain_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(pint.Quantity),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: (float(instance.m), str(instance.u))
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Use the same schema that would be used for the tuple
        return handler(_pint_base_repr)


_length_dim = ureg.meter.dimensionality
_angle_dim = ureg.radian.dimensionality
_pixel_dim = ureg.pixel.dimensionality


def _make_handler(dimensionality: str):
    def is_matching(
                q: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
            ) -> pint.Quantity:
        # Ensure target type
        if isinstance(q, pint.Quantity):
            pass
        elif isinstance(q, Sequence):
            m, u = q
            # Turn into Quantity: measure * unit
            q = m * ureg(u)
        else:
            raise ValueError(f"Don't know how to interpret type {type(q)}.")
        # Check dimension
        if not q.check(dimensionality):
            raise DimensionError(f"Expected dimensionality {dimensionality}, got quantity {q}.")
        # Return target type
        return q

    return is_matching


Length = Annotated[
    pint.Quantity, PintAnnotation, WrapValidator(_make_handler(_length_dim))
]
Angle = Annotated[
    pint.Quantity, PintAnnotation, WrapValidator(_make_handler(_angle_dim))
]
Pixel = Annotated[
    pint.Quantity, PintAnnotation, WrapValidator(_make_handler(_pixel_dim))
]


class Simple4DSTEMParams(BaseModel):
    '''
    Basic calibration parameters of a strongly simplified model
    of a 4D STEM experiment.

    See https://github.com/LiberTEM/Microscope-Calibration
    and https://arxiv.org/abs/2403.08538
    for the technical details.
    '''
    overfocus: Length
    scan_pixel_pitch: Length
    camera_length: Length
    detector_pixel_pitch: Length
    semiconv: Angle
    cy: Pixel
    cx: Pixel
    scan_rotation: Angle
    flip_y: bool
