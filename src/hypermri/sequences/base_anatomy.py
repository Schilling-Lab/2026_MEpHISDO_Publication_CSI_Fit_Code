from hypermri.utils.system.logging import init_default_logger

from ..core import MRIExperiment

logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


DEBUG_LOG = True  # Debug message flag
DEBUG_FILE = "/tmp/create_mask_debug.log"


class BaseAnatomy(MRIExperiment):
    """Experimental base class for anatomy classes."""

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts a pre-loaded MRIExperiment object as input."""
        # Extract MRIExperiment-specific kwargs and filter out others
        requested_load_data = kwargs.pop("load_data", False)
        log_mode = kwargs.pop("log_mode", "error")
        # Remove any sequence-specific kwargs that MRIExperiment doesn't understand
        kwargs.pop("reco_options", None)

        # Avoid re-triggering heavy data loading when study-level minimal/lazy modes pass strings
        load_data = (
            requested_load_data if isinstance(requested_load_data, bool) else False
        )

        # Call the parent (MRIExperiment) constructor with the path from the loaded object.
        # This ensures MRIExperiment's internal setup is done, but the data is already
        # loaded.
        super().__init__(str(mri_exp_obj.path), load_data=load_data, log_mode=log_mode)

        # Preserve the original handler if it has data loaded
        if hasattr(mri_exp_obj, "handler") and hasattr(mri_exp_obj.handler, "fid"):
            logger.debug("BaseAnatomy: preserving loaded handler from mri_exp_obj")
            self.handler = mri_exp_obj.handler
        elif hasattr(mri_exp_obj, "handler"):
            logger.debug("BaseAnatomy: handler exists but no fid, ensuring data load")
            self.handler = mri_exp_obj.handler
            # Ensure fid data is loaded if the handler exists but fid is missing
            if not hasattr(self.handler, "fid") or self.handler.fid is None:
                try:
                    self.handler.load_data(source=str(mri_exp_obj.path))
                    logger.debug(
                        f"BaseAnatomy: loaded data, handler has fid: {hasattr(self.handler, 'fid')}"
                    )
                except Exception as e:
                    logger.warning(f"BaseAnatomy: Failed to load data: {e}")

        # Copy data from the loaded MRIExperiment object using standardized access
        if (
            hasattr(mri_exp_obj, "primary_data")
            and mri_exp_obj.primary_data is not None
        ):
            self.data_array = mri_exp_obj.primary_data
        elif hasattr(mri_exp_obj, "data_array") and mri_exp_obj.data_array is not None:
            self.data_array = mri_exp_obj.data_array

        # Mirror handler parameters so metadata (e.g. center_ppm) stays available on this instance
        if hasattr(self.handler, "get_parameters"):
            try:
                handler_params = self.handler.get_parameters()
                if handler_params:
                    self.set_parameters(handler_params, allow_extra=True)
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.debug(f"BaseAnatomy: parameter synchronisation skipped: {exc}")

        # Initialize plotter using registry system
        try:
            from ..visualization.plotter_factory import create_plotter

            self.plotter = create_plotter(
                sequence_obj=self, category="anatomical", plotter_type="anatomy"
            )
        except ImportError:
            # Fallback to direct import if registry system not available
            from ..utils.plot.anatomyPlotter import AnatomyPlotter

            self.plotter = AnatomyPlotter(self)

        # try:
        #     self.extent = get_extent(data_obj=self)
        # except BaseException:
        #     self.extent = None

        # Initialize standardized data arrays (inherited from StandardizedMRIMixin via MRIExperiment)
        # self.data_array is already initialized via mixin (kspace_array no
        # longer stored)

        self.clear_lasso_patch_ref = []

    def _load_data_array(self):
        """Lazy loading implementation for scanner reconstructed data (seq2d)."""
        if hasattr(self, "handler") and self.handler is not None:
            try:
                return self.reconstruct(reco_type="seq2d")
            except Exception as e:
                logger.warning(f"Failed to lazy load data_array: {e}")
        return None

    def _load_reco_data(self):
        """Lazy loading implementation for raw reconstructed data."""
        if hasattr(self, "handler") and self.handler is not None:
            try:
                return self.reconstruct(reco_type="raw")
            except Exception as e:
                logger.warning(f"Failed to lazy load reco_data: {e}")
        return None


    def plot(self, *args, **kwargs):
        """
        Forwards the plotting call to the dedicated AnatomyPlotter instance.
        """
        return self.plotter.plot(*args, **kwargs)


