from pydantic import BaseModel, Field
from typing import Tuple, Optional
import json
import logging
import os

class Parameters(BaseModel, frozen=True):
    """
    Class to handle simulations of parameters.
    """
    #flat_spectrum.txt  dirac_spectrum.txt   gaia_055000450000.txt t5700g440a00x10p00_w1001_1031_combined.dat
    input_file: str = "gaia_055000450000.txt"
    input_resolving_power: int = Field(300000, ge=1)  # Modify acoording to the spectra we use as input.
    num_output_spectra: int = Field(1, ge= 1)

    convert_units: bool = True 
    
    apply_convolution: bool = True
    plot_convolution: bool = True  
    resolving_power: int = Field(11500, ge= 1)
    fwhm_wavelength: float = Field(0.75, gt=0)

    apply_resampling: bool = True
    plot_resampling: bool = True
    fixed_pixel_size: bool = False # If True, use the manually set pixel_size instead of computing it
    pixel_size: float = Field(0.245, gt=0) #Gaia pixel size

    plot_rescaling: bool = True
    reference_range: Optional[Tuple[float, float]] = None # Set to None will use automatic central part max flux.

    apply_trimming: bool = True
    trimming_range: Optional[Tuple[float, float]] = None #could also out a range if wanted (8560, 8600)
    plot_trimming: bool = True
   

    plot_noise: bool = True
    snr: float = Field(100, gt=0)

    def save_to_json(self, filename="config.json"):
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                json.dump(self.model_dump(), f, indent=4)
            logging.info(f"[save_to_json] Parameters saved to '{filename}'")

    @classmethod
    def load_from_json(cls, filename="config.json"):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            logging.info(f"Parameters loaded from '{filename}'")
            return cls(**data) #instance like in C
        except FileNotFoundError:
            logging.debug(f"Error: File '{filename}' not found.")
            return cls()
    
    def output_filename(self, index: int) -> str:
        import os
        base = os.path.splitext(os.path.basename(self.input_file))[0]
        return f"{base}_snr{int(self.snr)}_{index+1:04d}.txt"
