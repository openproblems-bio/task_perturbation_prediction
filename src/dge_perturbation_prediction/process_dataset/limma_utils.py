
import anndata as ad
import pandas as pd

import os
import subprocess
from tempfile import TemporaryDirectory


def limma_fit(
    adata: ad.AnnData,
    design: str,
    output_path: str,
    exec_path: str,
    Rscript_path: str,
    plot_output_path: str = None,
    verbose: bool = False
):
    """
    Fit a limma model to transcriptomic data, and save to a file for future use.

    Parameters
    ----------
    adata: anndata.AnnData
        Input AnnData object with raw pseudobulked counts in .X, and model covariates as columns in .obs
    design: str
        Design formula for the limma fit. Terms correspond to columns of `adata.obs`. For example, to model
        perturbation effect, set to "~0+pert". To model perturbation effect while correcting for library,
        set to "~0+pert+library". N.B. By limma's design, terms beyond the first are modeled using a mean-reference
        setup with the reference being the first category in the .obs column. See here for a comprehensive description
        of formula and model matrix setup: https://f1000research.com/articles/9-1444
    output_path: str
        Path where an .rds object containing a fit limma model will be stored. This path will then be input
        to, and read by, the `limma_contrast` function to extract various contrasts.
    exec_path: str
        Location of the viash executable for limma. Points to the implementation in viash-components by default.
    plot_output_path: str
        If provided, stores a PDF of the diagnostic voom plot showing mean-variance relationship among genes at
        this location. See the limma user guide for interpretation notes.
    verbose: logical
        If True, print stdout and stderr from the limma script.
    Returns:
        No return value - writes the fit object to file.
    -------

    """

    temp_prefix = os.path.expanduser('~') + '/.tmp/limma/'

    if not os.path.exists(temp_prefix):
        os.makedirs(temp_prefix)

    if exec_path is None:
        raise NotImplementedError('Currently executable must be local')

    with TemporaryDirectory(prefix=temp_prefix) as tempdirname:
        data_dir = tempdirname + '/'

        # write to file so limma can read it
        adata.write_h5ad(data_dir + 'input.h5ad')

        input_path = data_dir + 'input.h5ad'

        cmd = [
            Rscript_path,
            exec_path,
            '--input_h5ad',
            input_path,
            '--design',
            design,
            '--fit_output_path',
            output_path,
        ]
        if plot_output_path is not None:
            cmd += ['--plot_output_path', plot_output_path]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if verbose:
            print(result.stderr)
            print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                'Error in limma \n Error message: {} \n \n stdout: {}'.format(result.stderr, result.stdout)
            )


def limma_contrast(
    fit_path: str,
    contrast: str,
    exec_path: str,
    Rscript_path: str,
    verbose: bool = False
):
    """
    Given a fit limma model (e.g. from `limma_fit`), compute DE statistics for a contrast of interest.
    See https://f1000research.com/articles/9-1444 and the limma user guide for tips on designing contrasts.

    Parameters
    ----------
    fit_path: str
        Path to an .rds file with a fit limma model, e.g. from `limma_fit`.
    contrast: str
        Contrast of interest between coefficients of the fit model. For example, 'pertBelinostat-pertDMSO'.
    exec_path: str
        Path to viash executable for limma. Defaults to the viash-components executable.
    verbose: logical
        If True, print stdout and stderr from the limma script.
    Returns: pd.DataFrame
        Dataframe with one row per gene and columns indicating DE statistics.
    -------

    """
    if exec_path is None:
        raise NotImplementedError('Currently executable must be local')

    temp_prefix = os.path.expanduser('~') + '/.tmp/limma/'

    if not os.path.exists(temp_prefix):
        os.makedirs(temp_prefix)

    with TemporaryDirectory(prefix=temp_prefix) as tempdirname:
        data_dir = tempdirname + '/'

        # compute the contrast and write results to temporary dir
        cmd = [
            Rscript_path,
            exec_path,
            '--input_fit',
            fit_path,
            '--contrast',
            contrast,
            '--contrast_output_path',
            data_dir + 'contrast_result.csv',
        ]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if verbose:
            print(result.stderr)
            print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                'Error in limma \n Error message: {} \n \n stdout: {}'.format(result.stderr, result.stdout)
            )
        # read and return the contrast data
        res = pd.read_csv(data_dir + 'contrast_result.csv')
        res = res.rename({res.columns[0]: 'gene'}, axis=1)

        return res