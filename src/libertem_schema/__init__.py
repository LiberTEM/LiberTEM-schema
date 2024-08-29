from pydantic import BaseModel
import pint


__version__ = '0.1.0.dev0'


# From https://github.com/hgrecco/pint/issues/1166#issuecomment-1116309404
class PintType:
    Q = pint.Quantity

    def __init__(self, q_check: str):
        self.q_check = q_check

    def __get_validators__(self):
        yield self.validate

    def validate(self, v, validation_info):
        q = self.Q(v)
        assert q.check(self.q_check), f"Dimensionality must be {self.q_check}"
        return q


Length = PintType("[length]")
Angle = PintType("")
Pixel = PintType("[printing_unit]")


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

    class Config:
        json_encoders = {
            pint.Quantity: str
        }
