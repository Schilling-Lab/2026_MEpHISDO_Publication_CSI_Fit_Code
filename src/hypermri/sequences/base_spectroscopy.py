

from ..analysis.spectroscopy.core import get_freq_axis
from ..core.experiment import MRIExperiment
from ..utils.system.logging import LOG_MODES, init_default_logger
from ..utils.time.temporal import calc_sampling_time_axis

try:
    from matplotlib import colormaps

    get_cmap = colormaps.get_cmap  # Matplotlib 3.10+
except ImportError:
    pass


# initialize logger
logger = init_default_logger(__name__)
logger.setLevel(LOG_MODES["Warning"])

DEBUG_LOG = True  # Enable debug logging for widget debugging
DEBUG_FILE = "/tmp/base_spectroscopy_debug.log"  # Use widget-specific debug file


class BaseSpectroscopy(MRIExperiment):
    """ "Base class for most spectroscopy classes.

    This class provides some basic functionality which can be used for most
    spectroscopy sequences. It is encouraged to inherit this class for custom
    spectroscopy sequence classes.

    Methods
    -------
    summary() : pd.Dataframe.Styler
        Prints a summary of all frequency related parameters.

    single_spec_linebroadening(...)
        Linebroadens a given spectrum by LB Herz and returns it.
        See method docs for more info.

    single_spec_find_peaks(...)
        Finds peaks of a given spec using scipy.signal.find_peaks().
        See method docs for more info.

    single_spec_plot(...):
        Plots a single given spectrum. See method docs for more info.
    """

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts a pre-loaded MRIExperiment object as input."""
        # Extract sequence-specific kwargs
        kwargs.pop("load_data", False)
        kwargs.pop("log_mode", "error")
        kwargs.pop("reco_options", {})

        # CRITICAL FIX: Don't call super().__init__() with new path!
        # Instead, copy all attributes from the loaded MRIExperiment object

        # Copy all essential attributes from the loaded MRIExperiment
        self.path = mri_exp_obj.path
        self.handler = mri_exp_obj.handler  # This preserves the loaded data!
        self.reconstructions = getattr(mri_exp_obj, "reconstructions", {})

        # Copy data arrays if they exist
        if (
            hasattr(mri_exp_obj, "primary_data")
            and mri_exp_obj.primary_data is not None
        ):
            self.data_array = mri_exp_obj.primary_data
        elif hasattr(mri_exp_obj, "data_array") and mri_exp_obj.data_array is not None:
            self.data_array = mri_exp_obj.data_array
        else:
            self.data_array = None

        # Copy reco_data if it exists (from raw reconstructions)
        if hasattr(mri_exp_obj, "reco_data") and mri_exp_obj.reco_data is not None:
            self.reco_data = mri_exp_obj.reco_data

        # Copy other important attributes from MRIExperiment if they exist and are not None
        # Avoid setting reco_array to None as it triggers property setter validation
        if hasattr(mri_exp_obj, "reco_array") and mri_exp_obj.reco_array is not None:
            self.reco_array = mri_exp_obj.reco_array

        logger.debug(
            f"BaseSpectroscopy: Initialized with handler type {type(self.handler)}"
        )
        logger.debug(
            f"BaseSpectroscopy: Handler has fid: {hasattr(self.handler, 'fid')}"
        )
        if hasattr(self.handler, "fid") and self.handler.fid is not None:
            logger.debug(f"BaseSpectroscopy: Fid shape: {self.handler.fid.shape}")

        # Initialize mixins manually since we're not calling super().__init__()
        # Initialize standardized path handling
        super(
            MRIExperiment, self
        ).__init__()  # This calls StandardizedMRIMixin.__init__()
        # Set path again to ensure mixin is properly initialized
        self.path = mri_exp_obj.path

        # Keep standardized metadata in sync with the handler that still owns the data
        if hasattr(self.handler, "get_parameters"):
            try:
                handler_params = self.handler.get_parameters()
                if handler_params:
                    self.set_parameters(handler_params, allow_extra=True)
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.debug(
                    f"BaseSpectroscopy: parameter synchronisation skipped: {exc}"
                )

        # Initialize plotter using registry system
        try:
            from ..visualization.plotter_factory import create_plotter

            self.plotter = create_plotter(
                sequence_obj=self, category="spectroscopy", plotter_type="spectroscopy"
            )
        except ImportError:
            # Fallback to direct import if registry system not available
            from ..utils.plot.spectroscopyPlotter import SpectroscopyPlotter

            self.plotter = SpectroscopyPlotter(self)

        # Initialize standardized data arrays (inherited from StandardizedMRIMixin via MRIExperiment)
        # self.data_array is already initialized via mixin (kspace_array no
        # longer stored)

        # self.reco_rawdata()

    def _load_data_array(self):
        """Lazy loading implementation for scanner reconstructed data (seq2d)."""
        if hasattr(self, "handler") and self.handler is not None:
            try:
                return self.reconstruct(reco_type="seq2d")
            except Exception as e:
                logger.warning(f"Failed to lazy load data_array: {e}")
        return None

    def _load_reco_data(self):
        """Lazy loading implementation for raw reconstructed data with on-demand FID loading."""
        if hasattr(self, "handler") and self.handler is not None:
            # Check if FID data is available, if not try to load it on-demand
            if not hasattr(self.handler, "fid") or self.handler.fid is None:
                logger.info(
                    f"{self.__class__.__name__}: FID data not available, attempting on-demand loading..."
                )
                try:
                    # Load FID data on-demand
                    self.handler.load_data(source=None)
                    logger.info(
                        f"{self.__class__.__name__}: On-demand FID loading successful"
                    )
                except Exception as e:
                    logger.warning(
                        f"{self.__class__.__name__}: On-demand FID loading failed: {e}"
                    )
                    return None

            # Now try reconstruction if FID data is available
            if hasattr(self.handler, "fid") and self.handler.fid is not None:
                try:
                    return self.reconstruct(reco_type="raw")
                except Exception as e:
                    logger.warning(f"Failed to lazy load reco_data: {e}")
            else:
                logger.warning(
                    f"{self.__class__.__name__}: No FID data available for reconstruction"
                )
        return None

    @property
    def sampling_time_axis(self):
        # Prefer pre-calculated axis from handler if available (e.g. for MGE/EPSI)
        handler_axis = getattr(self.handler, "sampling_time_axis_s", None)
        if handler_axis is not None:
            return handler_axis

        return calc_sampling_time_axis(
            data_obj=self, npoints=self.n_acquisition - self.cut_off
        )

    @property
    def time_axis(self):
        return calc_sampling_time_axis(data_obj=self, start_with_0=True)

    @property
    def ppm(self):
        return get_freq_axis(
            data_obj=self, unit="ppm", cut_off=self.cut_off, npoints=None
        )

    @property
    def Hz(self):
        return get_freq_axis(
            data_obj=self, unit="Hz", cut_off=self.cut_off, npoints=None
        )


    def plot(self, *args, **kwargs):
        """
        Forwards the plotting call to the dedicated SpectroscopyPlotter instance.
        """
        return self.plotter.plot(*args, **kwargs)
