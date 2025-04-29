from pydantic import BaseModel, Field
from typing import Tuple
import json
import logging

class Parameters(BaseModel, frozen=True):
    """
    Class to handle simulations of parameters.
    """
    #flat_spectrum.txt  dirac_spectrum.txt   gaia_055000450000.txt
    input_file: str = "gaia_055000450000.txt"
    input_resolving_power: int = Field(300000, ge=1)  # Modify acoording to the spectra we use as input.
    num_output_spectra: int = Field(2, ge= 1)

    convert_units: bool = True 
    
    apply_convolution: bool = True
    plot_convolution: bool = True  
    resolving_power: int = Field(11500, ge= 1)
    fwhm_wavelength: float = Field(0.75, gt=0)

    apply_resampling: bool = True
    plot_resampling: bool = True
    pixel_size: float = Field(0.25, gt=0)

    plot_rescaling: bool = True
    reference_range: Tuple[float, float] = (8550, 8650)

    apply_trimming: bool = True
    plot_trimming: bool = True
    trimming_range: Tuple[float, float] = (8460, 8700)
    trimming_margin: float = 1.0

    plot_noise: bool = True
    snr: float = Field(10, gt=0)

    def save_to_json(self, filename="config.json"):
        with open(filename, "w") as f:
            json.dump(self.model_dump(), f, indent=4)
        logging.info(f"Parameters saved to '{filename}'")

    @classmethod
    def load_from_json(cls, filename="config.json"):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            return cls(**data) #instance like in C
        except FileNotFoundError:
            logging.debug(f"Error: File '{filename}' not found.")

            return cls()
    
    def output_filename(self, index: int) -> str:
        import os
        base = os.path.splitext(os.path.basename(self.input_file))[0]
        return f"{base}_snr{int(self.snr)}_{index+1:04d}.txt"
