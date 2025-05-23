import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import logging
import os

class Spectrum:
    """
    Class to manage spectrum data and operations.
    """

    def __init__(self):
        """
        Initialize spectrum attributes.
        """
        self.wavelength = np.array([])
        self.flux = np.array([])
    
    def copy(self):
        """
        Create a copy of the current spectrum instance.
        Returns:
            Spectrum: A new instance with copied wavelength and flux data.
        """
        new_spectrum = Spectrum()
        new_spectrum.wavelength = self.wavelength.copy()
        new_spectrum.flux = self.flux.copy()
        
        return new_spectrum
    
    
    def load_spectrum(self, file_name):
        """
        Load input spectrum from a file and parse data.
        """
        try:
            data = np.loadtxt(file_name)
            self.wavelength = data[:, 0]
            self.flux = data[:, 1]
            logging.info(f"Loaded spectrum from '{file_name}' successfully.")
            
        except Exception as e:
            logging.error(f"[load_spectrum] Failed to load spectrum: {e}")

    def convert_units(self, parameters, target_unit="A"):
        """
        Convert the wavelength units if enabled in parameters.

        Args:
            parameters (Parameters): The parameters object containing configuration.
            target_unit (str): The target unit for wavelength conversion. Default is Ångströms ("A").
                               Supports conversion from nm to Ångströms.
        """
        mean_wavelength = np.mean(self.wavelength)
        if target_unit == "A":
            if mean_wavelength > 1000:
                logging.warning("Wavelengths appear to already be in Ångströms. Skipping conversion.")
                return self
            self.wavelength *= 10
            logging.info("Converted units from nm to Ångströms.")
        else:
            logging.warning(f"Unsupported target unit '{target_unit}'. No conversion applied.")

        return self

            
    def convolve_spectrum(self, parameters, apply_convolution=True, verbose=True):
        """
        Apply Gaussian convolution using parameters from the Parameters class.

        Args:
            parameters (Parameters): The parameters object containing resolving power and FWHM.
            apply_convolution (bool): If True, applies convolution. If False, skips it.
            verbose (bool): If True, prints debugging details.
        """
        original_spectrum = self.copy()
        initial_flux = np.sum(self.flux)

        delta_lambda = np.sqrt(
            (np.mean(self.wavelength) / parameters.resolving_power)**2 -
            (np.mean(self.wavelength) / parameters.input_resolving_power)**2
        )
        sigma_wavelength = delta_lambda / (2 * np.sqrt(2 * np.log(2)))
        lambda_step = np.mean(np.diff(self.wavelength))
        sigma_pixels = sigma_wavelength / lambda_step

        logging.info(f"Sigma wavelength units: {sigma_wavelength:.2f} Å, pixel units: {sigma_pixels:.2f}")
        
        convolved_flux = gaussian_filter1d(self.flux, sigma_pixels, mode='nearest')
        self.flux = convolved_flux

        final_flux = np.sum(self.flux)
        conservation_ratio = final_flux / initial_flux if initial_flux != 0 else 0

        logging.info("Gaussian convolution completed.")
        logging.info(f"Flux before & after convolution: {initial_flux:.3f} | {final_flux:.3f}")
        logging.info(f"[convolve_spectrum] Flux conservation ratio: {conservation_ratio:.6f}")
            
        if parameters.plot_convolution:
            self.plot(
                reference_spectrum=original_spectrum, 
                processed_label="Convolved Spectrum", processed_color="red", processed_linestyle="-", 
                original_label="Original Spectrum", original_color="pink", original_linestyle="--"
            )

        return self


    def resample_spectrum(self, parameters, verbose=True):
        """
        Resample the spectrum onto a new uniform wavelength grid based on pixel size.

        If parameters.fixed_pixel_size is True, the pixel size uses the fixed value in parameters.pixel_size
        Otherwise, it is calculated as: 
            pixel_size = mean_wavelength / (3 * resolving_power)
        Args:
            parameters (Parameters): Object containing simulation parameters, including pixel size.
            verbose (bool): If True, prints debugging information.
        """
        original_spectrum = self.copy()

        if parameters.fixed_pixel_size:
            pixel_size = parameters.pixel_size
            logging.info(f"Using fixed pixel size: {pixel_size:.4f} Å")
        else: 
            pixel_size = np.mean(self.wavelength) / (3 * parameters.resolving_power)
        
        new_wavelength_grid = np.arange(self.wavelength[0], self.wavelength[-1], pixel_size)


        interpolator = interp1d(self.wavelength, self.flux, kind="linear", bounds_error=False, fill_value=0)
        resampled_flux = interpolator(new_wavelength_grid)

        self.wavelength = new_wavelength_grid


        self.flux = resampled_flux

        logging.info(f"Resampling completed.")
        logging.info(f"First: {self.wavelength[0]:.2f} Å, Center:{np.mean(self.wavelength)} Å, Last: {self.wavelength[-1]:.2f} Å")
        logging.info(f"Pixel size: {parameters.pixel_size:.2f} Å, New grid points: {len(self.wavelength)}.")

        if parameters.plot_resampling:    
            self.plot(reference_spectrum=original_spectrum, 
                             processed_label="Resampled Spectrum", processed_color="blue", processed_linestyle="-",
                             original_label="Original Spectrum", original_color="lightblue", original_linestyle="-", original_linewidth=5,processed_linewidth=0.7)
            
            self.plot(reference_spectrum=original_spectrum, display_type="both",processed_color="blue", processed_linestyle="-",
                             original_label="Original Spectrum", original_color="lightblue", original_linestyle="-", original_linewidth=5,processed_linewidth=0.7, zoom=(10100, 10215))

        return self
    
    def plot(self, reference_spectrum=None, display_type="both", zoom=None,
                        processed_label="Processed Spectrum", processed_color="red", processed_linestyle="-",
                        processed_linewidth=1.0, processed_zorder=2,
                        original_label="Original Spectrum", original_color="pink", original_linestyle="--",
                        original_linewidth=1.0, original_zorder=1):
        """
        Plot the spectrum before and after processing (convolution, resampling, etc.).
        
        reference_spectrum (Spectrum or None): The original (unprocessed) spectrum to compare against.
        display_type (str): Choose which spectra to display:
                - 'None': No plot
                - 'processed': Only the processed spectrum (e.g., convolved or resampled)
                - 'both': Original and processed spectrum
            zoom (tuple, optional): A tuple (min_wavelength, max_wavelength) to zoom in on a specific region.

            # Plot styling parameters:
            processed_label (str): Label for the processed spectrum.
            processed_color (str): Color for the processed spectrum.
            processed_linestyle (str): Linestyle for the processed spectrum.
            processed_alpha (float): Transparency for the processed spectrum.

            original_label (str): Label for the original spectrum.
            original_color (str): Color for the original spectrum.
            original_linestyle (str): Linestyle for the original spectrum.
            original_alpha (float): Transparency for the original spectrum.
        """

        plt.figure(figsize=(15, 10))
        if display_type == "both" and reference_spectrum is not None:
            plt.plot(reference_spectrum.wavelength, reference_spectrum.flux, 
                     label=original_label, 
                     color=original_color, 
                     linestyle=original_linestyle,
                     linewidth=original_linewidth,
                     zorder=original_zorder)

        if display_type in ["both", "processed"]:
            plt.plot(self.wavelength, self.flux, 
                     label=processed_label, 
                     color=processed_color, 
                     linestyle=processed_linestyle,
                     linewidth=processed_linewidth,
                     zorder=processed_zorder)

        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Flux")
        plt.legend()
        plt.grid(True)
        plt.title("Spectrum Comparison")

        if zoom:
            plt.xlim(zoom)

        plt.show()        

    def rescale_flux(self, parameters, verbose=True):
        """
        Rescale the flux vertically based on a reference level.
        Selects the max flux in the central region if no range is provided.

        Args:
            parameters (Parameters): Contains the reference range and SNR.
            verbose (bool): If True, shows logs and plots.
        """
        if parameters.reference_range is not None:
            min_ref, max_ref = parameters.reference_range
            mask = (self.wavelength >= min_ref) & (self.wavelength <= max_ref)
            selected_flux = self.flux[mask]

            if selected_flux.size == 0:
                logging.error("No flux found in specified reference range: %.2f-%.2f Å", min_ref, max_ref)
                logging.error("Wavelength range: %.2f-%.2f Å", self.wavelength[0], self.wavelength[-1])
                return self
            reference_flux = np.percentile(selected_flux, 100)
        # Using 100th percentile for our synthetic spectra
        #For real spectra, percentile should depend on SNR.

        else:
            
            width = 0.2 * (self.wavelength[-1] - self.wavelength[0])  # 20% width
            min_central = np.mean(self.wavelength) - width / 2
            max_central = np.mean(self.wavelength) + width / 2
            mask = (self.wavelength >= min_central) & (self.wavelength <= max_central)
            selected_flux = self.flux[mask]

            if selected_flux.size == 0:
                logging.warning("Central region empty. Falling back to global max.")
                reference_flux = np.max(self.flux)
            else:
                reference_flux = np.max(selected_flux)

            logging.info("Selected central range: {:,.2f}–{:,.2f} Å".format(min_central, max_central))

        self.reference_flux = reference_flux
        self.flux = (self.flux / self.reference_flux) * (parameters.snr ** 2)

        if verbose:
            new_max_flux = np.max(self.flux)
            logging.info(f"Reference flux used: {self.reference_flux:.3f}")
            logging.info(f"Max flux after rescaling: {new_max_flux:.3f}")
            logging.info("Rescaling completed.")

        if parameters.plot_rescaling:
            self.plot(
                processed_label="Rescaled Spectrum",processed_color="indigo", processed_linestyle="-", display_type="processed")
        return self

        
    def radial_velocity_shift(self, parameters, verbose=True):
        if verbose:
            logging.info(f"Applying radial velocity shift")

    def resample_stochastic(self, parameters, verbose=True):
        """
        Resample the spectrum for stochastic processes.

        If trimming is enabled, apply a fixed wavelength window either from parameters
        or calculated based on ±1000 km/s Doppler shift to ensure SPADES2 compatibility.
        """
        if verbose:
            logging.info("Resampling spectrum for stochastic process")

        if parameters.apply_trimming:
            
            if parameters.trimming_range is not None:
                min_wavelength, max_wavelength = parameters.trimming_range
                logging.info(f"Using manual trimming range: {min_wavelength:.2f}–{max_wavelength:.2f} Å")
            else:
    
                margin = np.mean(self.wavelength) * 1000 / 299_792.458

                min_wavelength = self.wavelength[0] + margin
                max_wavelength = self.wavelength[-1] - margin

            mask = (self.wavelength >= min_wavelength) & (self.wavelength <= max_wavelength)
            trimmed_wavelength = self.wavelength[mask]
            trimmed_flux = self.flux[mask]

            if trimmed_wavelength.size == 0 or trimmed_flux.size == 0:
                logging.error(f"No data found in trimming range {min_wavelength}-{max_wavelength} Å.")
                return self

            original_spectrum = self.copy()

            self.wavelength = trimmed_wavelength
            self.flux = trimmed_flux

            logging.info(f"Trimmed wavelength range: {self.wavelength[0]:,.2f} Å to {self.wavelength[-1]:,.2f} Å")
            logging.info(f"Number of points after trimming: {len(self.wavelength)}")

            if parameters.plot_trimming:
                self.plot(
                    reference_spectrum=original_spectrum,
                    processed_label="Trimmed Spectrum", processed_color="teal", processed_linestyle="-",
                    original_label="Original Spectrum", original_color="lightgray", original_linestyle="--",
                    display_type="both"
                )

        return self



    def generate_noise(self, parameters, verbose=True):
        """
        Add Poisson-distributed noise to the spectrum based on the signal-to-noise ratio (SNR).

        Args:
            parameters (Parameters): Contains SNR and reference wavelength range.
            verbose (bool): If True, prints debug information and plots before/after noise application.
        """
        original_spectrum = self.copy()
        
        self.flux = np.random.poisson(self.flux)

        if verbose:
            measured_mean = np.mean(self.flux)
            measured_std = np.std(self.flux)
            estimated_snr = measured_mean / measured_std if measured_std > 0 else float("inf")

            logging.info(f"Noise Generation Completed.")
            logging.info(f"Mean: {measured_mean:,.2f}, std: {measured_std:,.2f}, estimated snr : {estimated_snr:.2f}")

        if parameters.plot_noise:
            self.plot(
                reference_spectrum=original_spectrum,
                processed_label="Noisy Spectrum", processed_color="darkgreen", processed_linestyle="-", processed_linewidth=0.8,
                original_label="Ideal Spectrum", original_color="lightgreen", original_linestyle="-", original_linewidth=2.0
            )
        return self

    def save_spectrum(self, base_name: str, snr: float, index: int):
        """
        Save the processed spectrum to a file.

        Args:
            output_file (str): Name of the output file.
        """
        if self.wavelength.size == 0 or self.flux.size == 0:
            raise ValueError("No spectrum data to save. Ensure spectrum is processed before saving.")
    
        snr_folder = os.path.join("outputs", f"SNR{int(snr)}")
        filename = f"{base_name}_snr{int(snr)}_{index:04d}.txt"
        full_path = os.path.join(snr_folder, filename)

        os.makedirs(snr_folder, exist_ok=True)
        np.savetxt(full_path, np.column_stack((self.wavelength, self.flux)))

        logging.info(f"Spectrum saved to '{full_path}'.")