###############################################################################
#
# Copyright (C) 2022, Michele Cappellari
# E-mail: michele.cappellari_at_physics.ox.ac.uk
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
###############################################################################

import numpy as np

from ppxf import ppxf_util as util

###############################################################################

class ssp_lib:
    """
    This class is meant as an example that can be easily adapted by the users
    to deal with other spectral templates libraries, different IMFs or different
    chemical abundances.

    Input Parameters
    ----------------

    filename:
        Name of a Numpy np.savez() file containing the following arrays for a
        given SSP models library, like FSPS, Miles, GALEXEV, BPASS,...

        1. templates[npixels, n_ages, n_metals]
        2. lam[npixels] in Angstrom in common to all the spectra (can be non-uniform)
        3. fwhm[npixels] or scalar in Angstrom, for the instrumental line-spread function
        4. ages[n_ages] for the spectra along the 2nd dimension
        5. metals[n_metals] for the spectra along the 3nd dimension
        6. masses[n_ages, n_metals] mass of living stars + remnants for one SSP
        7. lums[n_ages, n_metals] luminosity of each spectrum in the r-band

        This file can be created with a command like::

            np.savez_compressed(filename, templates=templates,
                        masses=masses, lums=lums, lam=lam,
                        ages=ages, metals=metals, fwhm=fwhm)

    velscale:
        desired velocity scale for the output templates library in km/s 
        (e.g. 60). This is generally the same or an integer fraction of the 
        ``velscale`` of the galaxy spectrum used as input to ``ppxf``.
    FWHM_gal: 
        scalar or vector with the FWHM of the instrumental resolution of the 
        galaxy spectrum in Angstrom at every pixel of the stellar templates.
        
        - If ``FWHM_gal=None`` (default), no convolution is performed.

    Optional Keywords
    -----------------

    age_range: array_like with shape (2,)
        ``[age_min, age_max]`` optional age range (inclusive) in Gyr for the 
        MILES models. This can be useful e.g. to limit the age of the templates 
        to be younger than the age of the Universe at a given redshift.
    metal_range: array_like with shape (2,)
        ``[metal_min, metal_max]`` optional metallicity [M/H] range (inclusive) 
        for the MILES models (e.g.`` metal_range = [0, np.inf]`` to select only
        the spectra with Solar metallicity and above).
    norm_range: array_like with shape (2,)
        A two-elements vector specifying the wavelength range in Angstrom 
        within which to compute the templates normalization
        (e.g. ``norm_range=[5070, 5950]`` for the FWHM of the V-band).

        If ``norm_range=None`` (default), the templates are not normalized
        individually, but instead are all normalized by the same scalar, given
        by the median of all templates.

    norm_type: {'mean', 'max', 'lbol'} optional
        * 'mean': the templates are normalized to ``np.mean(template[band]) = 1``
          in the given ``norm_range`` wavelength range. When this keyword is
          used, ``ppxf`` will output light weights, and ``mean_age_metal()``
          will provide light-weighted stellar population quantities.
        * 'max':  the templates are normalized to ``np.max(template[band]) = 1``.
        * 'lbol':  the templates are normalized to ``lbol(template[band]) = 1``,
          where ``lbol`` is the integrated luminosity in the given wavelength
          range. If ``norm_range=[-np.inf, np.inf]`` and the templates extend
          over a wide wavelength range, the normalization approximates the
          true bolometric luminosity.

        One can use the output attribute ``.flux`` to convert light-normalized
        weights into mass weights, without repeating the ``ppxf`` fit.
        However, when using regularization in ``ppxf`` the results will not
        be identical. In fact, enforcing smoothness to the light-weights is
        not quite the same as enforcing it to the mass-weights.
    wave_range: array_like with shape (2,)
        A two-elements vector specifying the wavelength range in Angstrom for
        which to extract the stellar templates. Restricting the wavelength
        range of the templates to the range of the galaxy data is useful to
        save some computational time. By default ``wave_range=[3541, 1e4]``

    Output Parameters
    -----------------

    Stored as attributes of the ``ssp`` class:

    .ages_grid: array_like with shape (n_ages, n_metals)
        Age in Gyr of every template.
    .flux: array_like with shape (n_ages, n_metals)
        If ``norm_range is not None`` then ``.flux`` contains the mean flux
        in each template spectrum within ``norm_range`` before normalization.

        When using the ``norm_range`` keyword, the weights returned by 
        ``ppxf`` represent light contributed by each SSP population template.
        One can then use this ``.flux`` attribute to convert the light weights
        into fractional masses as follows::

            pp = ppxf(...)                                  # Perform the ppxf fit
            light_weights = pp.weights[~gas_component]      # Exclude gas templates weights
            light_weights = light_weights.reshape(reg_dim)  # Reshape to a 2D matrix
            mass_weights = light_weights/miles.flux         # Divide by this attribute
            mass_weights /= mass_weights.sum()              # Normalize to sum=1

    .ln_lam_temp: array_like with shape (npixels,)
        Natural logarithm of the wavelength in Angstrom of every pixel.
    .lam_temp: array_like with shape (npixels,)
        Wavelength in Angstrom of every pixel of the output templates.
    .metals_grid: array_like with shape (n_ages, n_metals)
        Metallicity [M/H] of every template.
    .n_ages: 
        Number of different ages.
    .n_metal: 
        Number of different metallicities.
    .templates: array_like with shape (npixels, n_ages, n_metals)
        Array with the spectral templates.

    """

    def __init__(self, filename, velscale, fwhm_gal=None, age_range=None,
                 metal_range=None, norm_range=None, norm_type='mean', wave_range=None):

        assert norm_type in ['max', 'lbol', 'mean'], "`norm_type` must be in ['max', 'lbol', 'mean']"

        a = np.load(filename)
        spectra, masses, lums, ages, metals, lam, fwhm_tem = \
            a["templates"], a["masses"], a["lums"], \
            a["ages"], a["metals"], a["lam"], a["fwhm"]

        metal_grid, age_grid = np.meshgrid(metals, ages)

        if fwhm_gal is not None:
            sigma = np.sqrt(fwhm_gal**2 - fwhm_tem**2)/np.sqrt(4*np.log(4))
            spectra = util.varsmooth(lam, spectra, sigma)

        templates, ln_lam_temp = util.log_rebin(lam, spectra, velscale=velscale)[:2]
        lam_temp = np.exp(ln_lam_temp)

        if norm_range is None:
            flux = np.median(templates[templates > 0])  # Single factor for all templates
            flux = np.full(templates.shape[1:], flux)
        else:
            assert len(norm_range) == 2, 'norm_range must have two elements [lam_min, lam_max]'
            band = (norm_range[0] <= lam_temp) & (lam_temp <= norm_range[1])
            if norm_type == 'mean':
                flux = templates[band].mean(0)          # Different factor for every template
            elif norm_type == 'max':
                flux = templates[band].max(0)                 # Different factor for every template
            elif norm_type == 'lbol':
                lbol = (templates[band].T*np.gradient(lam_temp[band])).T.sum(0)     # Bolometric luminosity in Lsun
                flux = lbol*(templates[band].mean()/lbol.mean())                    # Make overall mean level ~1
        templates /= flux

        if age_range is not None:
            w = (age_range[0] <= ages) & (ages <= age_range[1])
            templates = templates[:, w, :]
            age_grid = age_grid[w, :]
            metal_grid = metal_grid[w, :]
            flux = flux[w, :]

        if metal_range is not None:
            w = (metal_range[0] <= metals) & (metals <= metal_range[1])
            templates = templates[:, :, w]
            age_grid = age_grid[:, w]
            metal_grid = metal_grid[:, w]
            flux = flux[:, w]

        self.templates_full = templates
        self.ln_lam_temp_full = ln_lam_temp
        self.lam_temp_full = lam_temp
        if wave_range is not None:
            good_lam = (lam_temp >= wave_range[0]) & (lam_temp <= wave_range[1])
            ln_lam_temp = ln_lam_temp[good_lam]
            templates = templates[good_lam]

        self.templates = templates
        self.ln_lam_temp = ln_lam_temp
        self.lam_temp = lam_temp
        self.age_grid = age_grid    # in Gyr
        self.metal_grid = metal_grid
        self.n_ages, self.n_metals = age_grid.shape
        self.flux = flux
        self.mass_no_gas_grid = masses
        self.lum_grid = lums


###############################################################################

    def mass_to_light(self, weights, band="r", quiet=False):
        """
        Computes the M/L in a chosen band, given the weights produced in output by pPXF.

        :param weights: pPXF output with dimensions weights[miles.n_ages, miles.n_metal]
        :param band:
        :param quiet: set to True to suppress the printed output.
        :return: mass_to_light in the given band

        """
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        # This is eq.(2) in Cappellari+13
        # http://adsabs.harvard.edu/abs/2013MNRAS.432.1862C
        mlpop = np.sum(weights*self.mass_no_gas_grid)/np.sum(weights*self.lum_grid)

        if not quiet:
            print(f'(M*/L)_{band}: {mlpop:#.4g}')

        return mlpop


###############################################################################

    def plot(self, weights, nodots=False, colorbar=True, **kwargs):

        assert weights.ndim == 2, "`weights` must be 2-dim"
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        xgrid = np.log10(self.age_grid) + 9
        ygrid = self.metal_grid
        util.plot_weights_2d(xgrid, ygrid, weights,
                             nodots=nodots, colorbar=colorbar, **kwargs)


##############################################################################

    def mean_age_metal(self, weights, quiet=False):

        assert weights.ndim == 2, "`weights` must be 2-dim"
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        lg_age_grid = np.log10(self.age_grid) + 9
        metal_grid = self.metal_grid

        # These are eq.(1) and (2) in McDermid+15
        # http://adsabs.harvard.edu/abs/2015MNRAS.448.3484M
        mean_lg_age = np.sum(weights*lg_age_grid)/np.sum(weights)
        mean_metal = np.sum(weights*metal_grid)/np.sum(weights)

        if not quiet:
            print('Weighted <lg_age> [yr]: %#.3g' % mean_lg_age)
            print('Weighted <[M/H]>: %#.3g' % mean_metal)

        return mean_lg_age, mean_metal


##############################################################################
