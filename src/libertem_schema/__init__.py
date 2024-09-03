from typing import Any, Tuple
from numbers import Number

from typing_extensions import Annotated
from pydantic_core import core_schema
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidationError,
)
from pydantic.json_schema import JsonSchemaValue
import pint


__version__ = '0.1.0.dev0'

ureg = pint.UnitRegistry()


class DimensionError(ValueError):
    pass


def _get_annotation(reference: pint.Quantity):
    
    dimensionality = reference.dimensionality

    class Annotation:
        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source_type: Any,
            _handler: GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
            def validate_from_tuple(value: Tuple[Number, str]) -> pint.Quantity:
                m, u = value
                quantity = m * ureg(u)
                result = pint.Quantity(quantity)
                print("debug", quantity, dimensionality)
                if not result.check(dimensionality):
                    raise DimensionError(
                        f"Dimensionality mismatch: Type {type(result)} expected {dimensionality}."
                    )
                return result

            from_tuple_schema = core_schema.chain_schema(
                [
                    core_schema.tuple_positional_schema(items_schema=[
                        core_schema.float_schema(),
                        core_schema.str_schema()
                    ]),
                    core_schema.no_info_plain_validator_function(validate_from_tuple),
                ]
            )

            return core_schema.json_or_python_schema(
                json_schema=from_tuple_schema,
                python_schema=core_schema.union_schema(
                    [
                        # check if it's an instance first before doing any further work
                        core_schema.is_instance_schema(pint.Quantity),
                        from_tuple_schema,
                    ]
                ),
                serialization=core_schema.plain_serializer_function_ser_schema(
                    lambda instance: (float(instance.m), str(instance.u))
                ),
            )
    return Annotation


Length = Annotated[
    pint.Quantity, _get_annotation(ureg.meter)
]
Angle = Annotated[
    pint.Quantity, _get_annotation(ureg.radian)
]
Pixel = Annotated[
    pint.Quantity, _get_annotation(ureg.pixel)
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
