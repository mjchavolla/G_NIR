from pydantic import BaseModel, Field
from typing import Tuple
import json

class Parameters(BaseModel, frozen=True):
    """
    Class to handle simulations of parameters.
    """
    convert_units: bool = True 
    apply_convolution: bool = True
    plot_convolution: bool = True  

    apply_resampling: bool = True
    plot_resampling: bool = True  

    apply_rescaling: bool = True
    plot_rescaling: bool = True 

    apply_noise: bool = True  
    plot_noise: bool = True  

    
    num_output_spectra: int = Field(1, ge= 1)
    resolving_power: int = Field(11500, ge= 1)
    pixel_size: float = Field(0.25, gt=0)
    resolving_power: int = 11500
    fwhm_wavelength: float = Field(0.75, gt=0)

    reference_range: Tuple[float, float] = (8550, 8560)
    snr: float = Field(50, gt=0)

    def save_to_json(self, filename="config.json"):
        with open(filename, "w") as f:
            json.dump(self.model_dump(), f, indent=4)
        print(f"[save_to_json] Parameters saved to '{filename}'")


    @classmethod
    def load_from_json(cls, filename="config.json"):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            return cls(**data) #instance like in C
        except FileNotFoundError:
            print(f"[load_from_json] Error: File '{filename}' not found.")
            return cls()
        


