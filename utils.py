"""Helper utility functions and definitions for MPS_severity_forecasting.py
"""

#####################################
### Initial imports
#####################################

import pandas as pd
import numpy as np
import os, itertools
import scipy
from scipy.stats._distn_infrastructure import rv_discrete
from scipy.special import logit, expit
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import pystan
from joblib import dump, Parallel, delayed
from pathlib import Path
from copy import copy
import arviz as az
from typing import List, Callable, Optional, Dict
import logging

from config import *
for d in odir:
    Path(odir[d]).mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=odir['data'] + 'MPS_severity_forecasting.log', 
                    encoding='utf-8', level=logging.INFO)


plt.ion()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Use a Calibri-like font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'carlito'


#####################################
### Load and prepare data
#####################################

## Load data 
#### Cases
data = pd.read_csv(idir + case_file)
data.index = data.Date

#### Population -- Historical
data_pop_hist = pd.read_excel(
    idir + 'USPop_World_Development_Indicators.xlsx', 
    sheet_name = 'Data', nrows=1).T
data_pop_hist.index = [c.split(' [')[0] for c in data_pop_hist.index]
data_pop_hist_f = data_pop_hist.loc[[str(c) for c in range(1976, 2019)],0]
data_pop_hist_f.index = data_pop_hist_f.index.astype(int)
data_pop_hist_f.name = 'Historical'

#### Population -- Projections
data_pop_fut = pd.read_excel(
    idir + 'USCensus_np2017-t1.xlsx', 
    sheet_name = 'Main series', skiprows=range(5), nrows=46)
data_pop_fut.drop(axis=0, index=0, inplace=True) # Drop header row
data_pop_fut.index = data_pop_fut.Year.astype(int)
data_pop_fut_f = data_pop_fut['Population'] * 1000
data_pop_fut_f.name = 'Projection'

#### Population -- Merged
data_pop_m = data_pop_fut_f.loc[max(data_pop_hist_f.index)+1:].append(data_pop_hist_f).sort_values()


## Forecast assumptions - Rate per 100MM
data_MPSforecast = pd.DataFrame({
    '5yr': [1.31, 1.79, 0.82],
    '10yr': [1.30, 1.67, 1.02],
    '20yr': [1.29, 1.57, 1.11],
    }, index=['Status Quo', 'Pessimistic', 'Optimistic'])

samp_years = np.arange(max_year+1, max_year_sim, 1)
data_MPSforecast_table = data_MPSforecast.values[:,:,np.newaxis] * (data_pop_m.loc[samp_years].values / 1e8).astype(float)

Nsim_scenarios = data_MPSforecast_table.shape[0]
Nsim_windows = data_MPSforecast_table.shape[1]
Nsim_years = data_MPSforecast_table.shape[2]


#####################################
### Scipy distribution functions
#####################################

class gzipf_gen(rv_discrete):
    r"""A Zipf discrete random variable with user-definable q.
    
    Notes
    -----
    The probability mass function for `zipf` is:
    .. math::
        f(k, a) = \frac{1}{\zeta(a,q) k^a}
    for :math:`k \ge 1`.
    `zipf` takes :math:`a` as shape parameter. :math:`\zeta` is the
    zeta function (`scipy.special.zeta`) with offset :math:`q`
    """
    def _argcheck(self, alpha, q):
        return alpha > 0
        return q > 0

    def _pmf(self, k, alpha, q):
        Pk = 1.0 / scipy.special.zeta(alpha, q) / k**(alpha)
        return Pk

def zipf_dist(alpha, y_min):
    """A zipf distribution with user-definable q.
    
    WARNING: For numerical efficiency, this distribution is defined with 
    support up to b=10000. If you anticipate calculating this for probability 
    distributions that have significant probability at b>10000, you should adjust 
    this.
    """
    gzipf = gzipf_gen(a=y_min, b=10000, name='gen zipf', longname='Generalized Zipf')
    return gzipf(alpha=alpha, q=y_min)

def ltrunc_pdf(x, pdf, loc, tval, scale, offset, *args):
    """Truncates a probability distribution method of a scipy distribution on the left
    at location loc.
    """
    if np.isscalar(x):
        if x < loc: y = tval
        else: y = (pdf(x, *args) - offset) / scale
    else:
        sel = x < loc
        y = np.zeros(len(x))
        y[~sel] = (pdf(x[~sel], *args) - offset) / scale
        y[sel] = tval
    return y

def left_truncate_scipy_dist(dist: scipy.stats._distn_infrastructure.rv_generic, loc: int):
    """This function modified a scipy distribution object to truncate its pdf and cdf methods
    on the left side at value loc.
    
    NOTE: it does not explicitly modify other methods of the distribution.
    """
    dist.old_pdf = dist.pdf
    dist.old_cdf = dist.cdf
    dist.pdf = lambda *args: ltrunc_pdf(args[0], 
                                        dist.old_pdf, 
                                        loc, 
                                        np.nan, 
                                        (1-dist.old_cdf(loc, *args[1:])), 
                                        0, 
                                        *args[1:])
    dist.cdf = lambda *args: ltrunc_pdf(args[0], 
                                        dist.old_cdf, 
                                        loc, 
                                        0, 
                                        1-dist.old_cdf(loc, *args[1:]), 
                                        dist.old_cdf(loc, *args[1:]), 
                                        *args[1:])
    return dist


#####################################
### Scipy and Stan model definitions
#####################################

## Stan model location
## This is a dictionary with keys equal to the model name and values being a length-2 list,
## where the first element is a reference to a stan model file and the second being a list
## of associated source includes.
stan_models: Dict[str, List] = {
    'weibull': ['weibull.stan', []],
    'pareto': ['discrete_pareto.stan', [os.getcwd()+'/hurwitz_zeta.hpp']],
    'lognormal': ['lognormal.stan', []],
    }

## Scipy model definitions
## This is a dictionary with keys equivalent to the keys of stan_models and values being scipy
## distribution instances.
scipy_models: Dict[str, Callable[..., scipy.stats._distn_infrastructure.rv_generic]] = {
    'weibull': lambda ext_data, i_sim, y_min: left_truncate_scipy_dist(scipy.stats.weibull_min(
                    ext_data['alpha'][i_sim], scale=ext_data['sigma'][i_sim]),
                    loc = y_min),
    'pareto': lambda ext_data, i_sim, y_min: zipf_dist(alpha=ext_data['alpha'][i_sim], y_min=y_min),
    'lognormal': lambda ext_data, i_sim, y_min: left_truncate_scipy_dist(scipy.stats.lognorm(ext_data['sigma'][i_sim], 
                    scale=np.exp(ext_data['mu'][i_sim])), loc=y_min)
    }

#####################################
### Diagnostic plotting functions
#####################################

def traceplot(ext_data, title='Traceplot'):
    """Generate a traceplot for a set of extracted Stan data samples
    """
    fig,axs = plt.subplots(len(ext_data), 1, figsize=(8,10), sharex='all', sharey='row')
    for i,par in enumerate(ext_data.keys()):
        try:
            if len(ext_data[par].shape) in [1,2]:
                axs[i].plot(ext_data[par]) #, color='.5')
            elif len(ext_data[par].shape) == 3:
                axs[i].plot(
                    ext_data[par].reshape(ext_data[par].shape[0], np.product(ext_data[par].shape[1:])), alpha=0.2)
            else:
                logging.warning("Not plotting trace for "+par+"; too many dimensions")
            
            for k in range(1, Nchains+1):
                axs[i].axvline(Niter/2 * k, c='b', zorder=-1, alpha=0.5)
        except:
            logging.warning("Traceplot plotting does not work for: ", par)
        axs[i].set_ylabel(par)

        axs[len(ext_data) - 1].set_xticks(np.arange(0, (Niter/2)*Nchains+1, Niter/2))

    axs[-1].set_xlabel('Simulation step')
    axs[0].set_title(title)

def loo_plot(loo_dic: Dict,
             elpd_type: str,
             **kwargs) -> None:
    """
    Plot a facet grid comparing LOO ELPD results across y_mins and
    fat_vars.
    
    Parameters
    ----------
    loo_dic: Dict
        Nested dictionary of arviz.loo outputs.  Dictionary should
        have nested keys of fat_var strings, y_min ints, elpd integration
        types (expected to be ['total', '>10'])
    elpd_type: str
        Should be one of ['total', '>10'].  This output will be picked
        out of loo_dic for plotting
    **kwargs are passed to plt.subplots
    
    Returns
    -------
    None
    """
    ## Lookup fat_vars and y_mins lists from the loo_dic dictionary keys
    fat_vars = list(loo_dic.keys())
    y_mins = list(loo_dic[fat_vars[0]].keys())
    ## Create plot
    fig, axs = plt.subplots(len(y_mins), len(fat_vars), **kwargs)
    plt.suptitle(f'Leave-one-out cross validation (PSIS-LOO; {elpd_type})')
    for i_var, fat_var in enumerate(fat_vars):
        for i_ymin, y_min in enumerate(y_mins):
            py = [np.random.normal(loo_dic[fat_var][y_min][model][elpd_type], loo_dic[fat_var][y_min][model][f'{elpd_type}_se'], 10000) 
                  for model in stan_models]
            axs[i_var,i_ymin].violinplot(py, np.arange(len(stan_models)), showextrema=False)
            axs[i_var,i_ymin].set_xticks(np.arange(len(stan_models)))
            axs[i_var,i_ymin].set_xticklabels(stan_models)
            if i_var==0: axs[0,i_ymin].set_title("$x_{{\\rm{{min}}}}={}$".format(y_min))
            if i_ymin==0: axs[i_var,0].set_ylabel(fat_var)

def loo_plot_per_x_min(loo_dic: Dict,
            y_min: int, 
            elpd_type: str) -> None:
    """Plot a row of facets comparing ELPD LOO results across fat_var's
    for a given y_min.
    
    Parameters
    ----------
    loo_dic: Dict
        Nested dictionary of arviz.loo outputs.  Dictionary should
        have nested keys of fat_var strings, y_min ints, elpd integration
        types (expected to be ['total', '>10'])
    y_min: int 
        y_min value to pick out from loo_dic for plotting.
    elpd_type: str
        Should be one of ['total', '>10'].  This output will be picked
        out of loo_dic for plotting
    
    Returns
    -------
    compare_elpd: np.ndarray
        N_fat_vars x N_models x N_models array of pairwise simulated model
        comparisons, i.e. probabilities above 50% that model j has highet ELPD than model i.
    """
    compare_elpd = np.zeros([len(fat_vars), len(stan_models), len(stan_models)])
    fig, axs = plt.subplots(1, len(fat_vars), sharey="col")
    for i_var, fat_var in enumerate(fat_vars):
        ## Use monte carlo to estimate ELPD distributions from normal approximation
        ## TODO use analytic comparison formulae from Vehtari et al. paper instead
        py = [np.random.normal(
                    loo_dic[fat_var][y_min][model][elpd_type], 
                    loo_dic[fat_var][y_min][model][f'{elpd_type}_se'], 
                    100000) 
                for model in stan_models]
        ## Compare distributions
        for i_model in range(len(stan_models)):
            for j_model in range(i_model+1, len(stan_models)):
                compare_elpd[i_var, i_model, j_model] = 0.5 - np.mean(py[j_model] > py[i_model])
        ## Plot
        axs[i_var].violinplot(py, np.arange(len(stan_models)), showextrema=False)
        axs[i_var].set_xticks(np.arange(len(stan_models)))
        axs[i_var].set_xticklabels(stan_models)
        axs[i_var].set_title(fat_var)
        max_preference = 50+int(np.abs(100*compare_elpd)[i_var].max())
        axs[i_var].text(0.1, 0.1, 
                           f'Max. preference: {max_preference}%',
                           transform=axs[i_var].transAxes)
    axs[0].set_ylabel("ELPD")
    plt.suptitle("$x_{{\\rm{{min}}}}={}$".format(y_min))
    
    return compare_elpd

def loo_plot_pointwise_likelihood(fit_dic: Dict,
                                  ext_dic: Dict,
                                  ) -> None:
    """Plot a facet grid across y_mins and fat_vars of the point-wise
    log likelihoods versus the event severity, x.
    
    Parameters
    ----------
    fit_dic, ext_dic: Dict
        Nested dictionary of pytan sampling fit result objects and extract() results
        of the same objects.  The nesting should have levels of fat_vars, y_mins, and models.
    
    Returns
    -------
    None
    """
    ## Lookup lists of y_mins, fat_vars, models
    fat_vars = list(fit_dic.keys())
    y_mins = list(fit_dic[fat_vars[0]].keys())
    models = list(fit_dic[fat_vars[0]][y_mins[0]].keys())
    ## Make plot
    fig, axs = plt.subplots(len(y_mins), len(fat_vars), sharey="all", sharex='all')
    for i_var, fat_var in enumerate(fat_vars):
        for i_ymin, y_min in enumerate(y_mins):
            for model in models:
                axs[i_var,i_ymin].plot(
                    fit_dic[fat_var][y_min][model].data['y'] , 
                    ext_dic[fat_var][y_min][model]['log_likelihood'].mean(axis=0),
                    '.', label=model)
            if i_var==0: axs[0,i_ymin].set_title("$x_{{\\rm{{min}}}}={}$".format(y_min))
            if i_ymin==0: axs[i_var,0].set_ylabel(fat_var)

    axs[0, 0].legend()
    for i_var in range(len(fat_vars)): axs[i_var, 0].set_ylabel('Pointwise-log likelihood')
    for i_ymin in range(len(y_mins)): axs[-1, i_ymin].set_xlabel('Severity')

#####################################
### Rare event probability functions
#####################################

def p_compound(x: np.ndarray) -> float:
    """Calculate the compound probability of a N-length vector of probabilities, x; i.e. the probability
    that at least one event will occur given N trials each with probability x_n.
    
    The calculation is done in log space for stability.
    """
    return 1. - np.exp(np.sum(np.log(1. - np.array(x))))

## Random Poisson draw for number of events
prv = lambda x: scipy.stats.poisson(x).rvs()

def calc_rare_event_prob(
        ext_data: Dict[str, np.ndarray], 
        scipy_model: Callable,
        fat_var: str='Fatalities', 
        target_fatal: int=60, 
        y_min: int=4,
        scenarios: List[int]=range(3), 
        windows: List[int]=range(3), 
        ): 
    """Do compound probability calculations to project the probability of
    rare events occuring over a future time period.
    
    Parameters
    ----------
    ext_data: Dict[str, np.ndarray]
        The dictionary of stan simulation results for the target model
    scipy_model: Callable
        Function that extracts parameters from ext_data and returns PDF of the tail model.
    fat_var: str
        y-axis label for the fatality variable
    target_fatal: 
        Rare event threshold to set lower bound of rare event probability integral.
    y_min: int
        Minimum definition of a mass shooting
    scenarios, windows: List[int]
        Indexes of scenarios and windows to extract from the global variable data_MPSforecast_table
    
    Returns
    -------
    Generates matplotlib figure and returns
    
    p_model_int: [N_MC] np.ndarray
        Integrated probability of exceeding the threshold target_fatal for each model draw.
    p_model_yr_int: np.ndarray
        Integrated probability of encountering an exceedence 
    p_model_yr_cum: np.ndarray
        Cumulative probability of encountering an exceedence from year 1 to N.
        Dimensions: N_{simulation samples} X N_scenarios x N_windows x N_years
    """
    logging.info(f"Calculating rare event probability for "
                 f"{fat_var}, {scipy_model.__name__}, {target_fatal}, {y_min}")
    ## Calculate
    p_model_int = np.zeros(int((Niter/2)*Nchains))
    p_model_yr_int = np.ones([int((Niter/2)*Nchains), len(scenarios), len(windows), Nsim_years])
    p_model_yr_cum = np.ones([int((Niter/2)*Nchains), len(scenarios), len(windows), Nsim_years])
    ## Calculate the probability that each simulated event belonds to the tail above the threshold
    ## NOTE: This is hardcoded to assume that there is a module-level variable, data, which is a
    ## pandas data frame containing the actual observed distribution of variable data. 
    ## We calculate this probabilist as a constant from the data, essentially fixing as known the
    ## probability in the left side of the data distribution.
    ## We do a quick assertion check to verify at least that the given data
    ## has the expected minimum value of 4 corresponding to a mass public shooting.
    assert data[fat_var].min() == 4
    C_t = 1 - np.mean(data[fat_var] < y_min)
    ## If target_fatal is the minimum value, we assert that the adjustment factor is unity
    if y_min == 4: assert C_t==1.
    logging.info(f"Setting C_t constant equal to {C_t} for y_min={y_min} based on data distribution")
    ## Step through scenarios and windows
    for i_scen in scenarios:
        for i_win in windows:
            ## This can be slow, so give the user an update
            logging.info(f"calculating...var: {fat_var}, scen: {i_scen}, win: {i_win}")
            for i_sim in range(int((Niter/2)*Nchains)):
                ## TODO may be better to caclulate and store these probabilities in
                ## log space for better precision.
                ## Integrate the fatality distribution to get the base probability
                p_model = scipy_model(ext_data, i_sim, y_min)
                ## Integrate probability from target_fatal to infinity
                p_model_int[i_sim] = C_t * (1 - p_model.cdf(target_fatal))
                ## Calculate individual probability by year
                sim_events = prv(data_MPSforecast_table[i_scen, i_win])
                for i_yr in range(Nsim_years):
                    p_model_yr_int[i_sim,i_scen,i_win,i_yr] = p_compound([p_model_int[i_sim],]*sim_events[i_yr])
                ## Calculate cumulative probability year by year
                p_model_yr_cum[i_sim,i_scen,i_win,0] = p_model_yr_int[i_sim,i_scen,i_win,0]
                for i_yr in range(1, Nsim_years):
                    p_model_yr_cum[i_sim,i_scen,i_win,i_yr] = p_compound([
                        p_model_yr_cum[i_sim,i_scen,i_win,i_yr-1], p_model_yr_int[i_sim,i_scen,i_win,i_yr]
                        ])
    return p_model_int, p_model_yr_int, p_model_yr_cum

def plot_rare_event_lines(
        p_model_yr_cum: np.ndarray,
        fat_var: str='Fatalities', 
        target_fatal: int=60, 
        y_min: int=4,
        scenarios: List[int]=range(3), 
        windows: List[int]=range(3), 
        prange: List[int]=[5,95],
        model_name: str=''
    ):
    """Plot forecasting probability calculations for rare events timeseries
    as lineplots
    
    Parameters
    ----------
    p_model_yr_cum: np.ndarray
        The result of calc_rare_event_prob; simulated event probabilities for 
        various scenarios, windows, and over time.
    fat_var: str
        y-axis label for the fatality variable
    target_fatal: 
        Rare event threshold to set lower bound of probability integral.
    y_min: int
        Minimum definition of a mass shooting
    scenarios, windows: List[int]
        Indexes of scenarios and windows to extract from the global variable data_MPSforecast_table
    prange: List[int]
        Probability range for error region, e.g. [5,95] for 90% interval
    model_name: str=''
        Model name to put in title
    
    Returns
    -------
    None
    """
    logging.info(f"Plotting rare event probability for "
                 f"{fat_var}, {model_name}, {target_fatal}, {y_min}")
    fig, axs = plt.subplots(1, len(windows), figsize=(3*len(windows), 4))
    if len(windows) == 1: axs = [axs]
    for axi in range(len(axs)):
        for i_scen,s in enumerate(data_MPSforecast.index[scenarios]):
            ps = np.percentile(p_model_yr_cum[:,i_scen,axi,:], [prange[0],50,prange[1]], axis=0) * 100
            axs[axi].plot(
                np.append([max_year], samp_years), np.append(0, ps[1]), 
                label=s, ls=['solid','dashed','dotted'][i_scen], color=colors[i_scen])
            axs[axi].fill_between(
                np.append([max_year], samp_years), 
                np.append(0, ps[0]), np.append(0, ps[2]), 
                lw=0, alpha=0.2, color=colors[i_scen], label=None)
        axs[axi].set_title(data_MPSforecast.columns[axi])
        axs[axi].grid()
    axs[0].set_ylabel(
        'Probability of shooting with >{} {}\nobserved since '\
            .format(target_fatal, fat_var)+str(max_year)+' (%)')
    axs[0].legend()
    plt.suptitle(model_name)


#####################################
### Probability distribution plotting functions
#####################################

ccdf_transform = lambda x, norm: norm * (1-np.nancumsum(x / np.nansum(x)))

def remove_vertical_line(ax, beginning=True):
    """Removes the vertical line at the beginning/end of a matplotlib
    histogram.  Use beginning=True to fix a cccdf plot, or beginning=False
    to fix a CDF plot.
    
    See https://stackoverflow.com/a/52921726
    """
    axpolygons = [poly for poly in ax.get_children() 
                  if isinstance(poly, matplotlib.patches.Polygon)]
    for poly in axpolygons:
        if beginning is True:
            poly.set_xy(poly.get_xy()[1:])
        else:
            poly.set_xy(poly.get_xy()[:-1])

def individual_dist_fit(ext_data: Dict[str, np.ndarray], 
                        scipy_model: scipy.stats._distn_infrastructure.rv_generic, 
                        actual_data: pd.Series,
                        hrange: List[float]=[4,1000], 
                        hbins: int=30, 
                        Ndraws: int=25, 
                        model_name: str='', 
                        xlab: str='Gunfire fatalities', 
                        inset: Optional[str]='alpha', 
                        inset_label: Optional[str]=None, 
                        prange: List[float]=[5,95], 
                        ylim: Optional[List[float]]=None, 
                        y_min: int=4, 
                        ax: Optional[plt.Axes]=None, 
                        color_i: int = 1, 
                        ls: str='-',
                        ccdf: bool=True,
                        show_actual: bool=True,
                        px_res: int=300,
                        dist_res: int=1000):
    """Plot a histogram of a probability distribution together with analytic forms of a fitted model.
    
    NOTE: if you set ccdf=True and you see surprising behavior in the plotted analytic functions, it's 
    probably because your hrange does not extend far enough into the tail to calculate the complementary 
    CDF with sufficient precision.  The reason for this is that we calculate the ccdf by mnumeric 
    integration over a grid of points bounded by hrange.
    
    Parameters
    ----------
    ext_data: Dict[str, np.ndarray]
        Extracted stan sample data.
    scipy_model: scipy.stats._distn_infrastructure.rv_generic
        Scipy distribution model to use to generate distribution analytic form.
    actual_data: pd.Series
        Actual observed event data to plot as histogram
    hrange: List[float]=[4,1000]
        Histogram plotting x-axis range
    hbins: int=30
        Number of histogram bins.  If plotting a ccdf, you will likely
        want to set this to a large number (e.g. 1000)
    Ndraws: int=25
        Number of draws from the modeled distribution to plot as spaghetti lines.
    model_name: str='', 
        Name of model to insert in legend
    xlab: str='Gunfire fatalities'
        X-axis label
    inset: Optional[str]='alpha'
        Variable from ext_data to visualize in inset plot
    inset_label: Optional[str]=None
        Label for inset plot.  If none, will be set to inset value.
    prange: List[float]=[5,95] 
        Posterior interval to plot for uncertainty visualization.
    ylim: List[float]=None
        y-axis range to plot. Since this will be a log-log plot, the starting element should be >0
        If None, the min will be set to [5e-5,5] and the max will be set to the greater of 1
        or the max of the plotted actual_data histogram (if generated).
    y_min: int=4
        Minimum x-axis value to plot for observed data and fitted models.
    ax: Optional[plt.Axes]=None
        Existing Axes object to plot onto; if None, will create a new Axes.
    color_i: int = 1
        Which color from the color wheel to use in coloring analyic line.
    ls: str='-'
        Linestyle for analytic distribution visualizations
    ccdf: bool=True
        If False, plot a PDF.  If False, plot a CCDF -- complemntary
        cumulative distribution function.
    show_actual: bool=True
        If False, set the actual plot histogram to not visible.  This is useful
        because ccdf plots rely on normalizing against the actual histogram, but
        in some cases (like if you're doing repeated overplotting) you may not want
        to show this multiple times.
    px_res: int=300
        Resolution (i.e. number of points) for calculating and plotting the analytic 
        model functions.
    dist_res: int=1000,
        Resolution (i.e. number of samples) for calculating the posterior pdf/ccdf
        of the analytic modeul functions
    
    Returns
    -------
    None
    """
    logging.info(f"Running individual_dist_fit on {model_name}")
    
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    
    if ccdf and hbins < 1000:
        logging.warning("You are plotting a ccdf plot with hbins < 1000; are you "
                        "certain you want a resolution that low?")
    
    if ylim is None: ylim=[5e-5, 1]
    
    ## Plot actual data
    if actual_data is not None:
        logging.info(f"Plotting actual data")
        bins = np.logspace(np.log10(hrange[0]), np.log10(hrange[1]), hbins)
        if ccdf is False:
            ## Use bar-style histograms, normalize on data > y_min
            counts, null = np.histogram(actual_data, bins=bins) #null result will be the same as bins
            sel = bins[:-1] >= y_min
            counts_n = counts / np.sum(counts[sel]) / (bins[1:] - bins[:-1]) # normalize
            ## Plot distribution at < y_min
            if y_min > hrange[0]:
                hist_n_all, hist_bins_all, hist_patches_all = ax.hist(
                        bins[:-1][~sel], 
                        bins=bins, 
                        weights=counts_n[~sel], 
                        cumulative=False,
                        density=False,
                        label='All events' if show_actual else None, 
                        log=True, zorder=1, color='0.6',
                        histtype='bar')
                ylim[1] = max(ylim[1], max(hist_n_all))
                if show_actual is False:
                    for patch in hist_patches_all: patch.set_visible(False)
            ## Plot distribution at >= y_min
            hist_n, hist_bins, hist_patches = ax.hist(
                    bins[:-1][sel], 
                    bins=bins[bins>=y_min], 
                    weights=counts_n[sel], 
                    cumulative=False,
                    density=False,
                    label=(f'Events (>{y_min})' if y_min > hrange[0] else 'Events') if show_actual else None, 
                    log=True, zorder=2, color=colors[0],
                    histtype='bar')
        else:
            ## When plotting the CCDF, use step-style histogram and pass
            ## the actual data rather than pre-calculated weights.
            if y_min > hrange[0]:
                hist_n_all, hist_bins_all, hist_patches_all = ax.hist(
                        actual_data, 
                        bins=bins, 
                        cumulative=-1,
                        density=True,
                        label='All events' if show_actual else None, 
                        log=True, zorder=1, color='0.6',
                        histtype='step')
                ylim[1] = max(ylim[1], max(hist_n_all))
                if show_actual is False:
                    for patch in hist_patches_all: patch.set_visible(False)
            ## Plot distribution at >= y_min
            hist_n, hist_bins, hist_patches = ax.hist(
                    actual_data, 
                    bins=bins[bins>=y_min], 
                    cumulative=-1,
                    density=True,
                    label=(f'Events (>{y_min})' if y_min > hrange[0] else 'Events') if show_actual else None, 
                    log=True, zorder=2, color=colors[0],
                    histtype='step',
                    lw=3)
        if show_actual is False:
            for patch in hist_patches: patch.set_visible(False)
        ylim[1] = max(ylim[1], max(hist_n))
        if ccdf: remove_vertical_line(ax)
    
    ## Setup x-values for plotting for either continuous or discrete models
    ## Note that for a ccdf we need to calculate far into the tail because the plot
    ## will be sensitive to the mass of probability calculated on the right tail.
    px_max = hrange[1]+1
    if hasattr(scipy_model(ext_data, 0, y_min).dist, 'pdf'):
        dist_type = 'continuous'
        px = np.logspace(np.log10(y_min), np.log10(px_max), px_res)
    else:
        dist_type = 'discrete'
        px = np.arange(y_min, px_max + 1)

    ## Calculate simulated distributions as either pdf, pmf, or ccdf
    logging.info(f"Calculating simulated distributions")
    dist_samples = np.random.randint(0, (Niter/2)*Nchains, dist_res)
    sim_dist = np.zeros([dist_res, len(px)])
    for ix, i in enumerate(dist_samples):
        if ccdf is False:
            if dist_type is 'continuous':
                sim_dist[ix] = scipy_model(ext_data, i, y_min).pdf(px)
            else:
                sim_dist[ix] = scipy_model(ext_data, i, y_min).pmf(px)
        else:
            if dist_type is 'continuous':
                sim_dist[ix] = 1 - scipy_model(ext_data, i, y_min).cdf(px)
            else:
                ## We shift the plotting axis by 1 so that the 0th point aligns to ccdf=1
                sim_dist[ix] = 1 - scipy_model(ext_data, i, y_min).cdf(px-1)
        
    ## To avoid plotting a vertical line at probability discontinuities
    sim_dist[sim_dist == 0] = np.nan
    
    ## Plot individual draws
    if Ndraws>0: logging.info(f"Plotting individual draws")
    for n in range(Ndraws): 
        samp = np.random.randint(0, dist_res)
        ax.plot(px, 
                sim_dist[samp], 
                color='0.5', alpha=0.3, zorder=4, lw=0.5,
            label=model_name+' fit (sample)' if n == 0 else None, )
    ## Plot overall distribution
    logging.info(f"Plotting analytic distributions")
    pps = np.percentile(sim_dist, [prange[0], 50, prange[1]], axis=0)
    ax.plot(px, 
            pps[1],
            label=model_name+' fit', lw=3, color=colors[color_i], zorder=5, ls=ls)
    ax.fill_between(px, 
                    pps[0], 
                    pps[2],
            label=None, alpha=0.3, color=colors[color_i], zorder=3, ls=ls)
    ax.set_ylabel('Complementary cumulative probability function' 
                        if ccdf else 'Probability density')
    ax.set_xlabel(xlab)
    ax.legend(loc = 'lower left')
    ax.loglog()
    ax.set_ylim(ylim)
    ax.set_xlim(hrange)
    
    if inset is not None:
        logging.info(f"Making inset plot")
        ax2 = plt.axes([.7, .7, .15, .15])
        ax2.hist(ext_data[inset], 50, color='0.5', density=True)
        ax2.set_xlabel(inset_label if inset_label is not None else inset)
        ax2.set_ylabel('$p$')
        ax2.axvline(np.mean(ext_data[inset]), color='r')
        for p in np.percentile(ext_data[inset], [5,95]):
            ax2.axvline(p, color='r', ls='dashed')
        ax2.xaxis.set_minor_locator(AutoMinorLocator())

def all_dist_fit(ext_datas, scipy_models, actual_data,
                        Ndraws=0, 
                        ylim=[1e-7,1], 
                        inset=None,
                        **kwargs):
    """Repeatedly call individual_dist_fit across a variety of models to plot multiple model
    distributions together.
    
    kwargs are passed to individual_dist_fit
    """
    assert ext_datas.keys() == scipy_models.keys()
    logging.info(f"Running all_dist_fit")
    
    ls_list = ['-', '--', ':', '-.']*10
    
    for m_i, model in enumerate(ext_datas):
        individual_dist_fit(
                ext_datas[model], 
                scipy_models[model],
                actual_data=actual_data,
                model_name=model,
                Ndraws=Ndraws, 
                ylim=ylim,
                inset=inset,
                color_i=m_i+1, 
                ls=ls_list[m_i],
                show_actual=True if m_i==0 else False,
                **kwargs)


#####################################
### Latex output helpers
#####################################

def format_prob(x: np.ndarray, 
                d: int=2, 
                r: List[int]=[5,95], 
                percent: bool=True, 
                thresh: float=1e-5,
                agg: Callable=np.mean):
    """Format a distribution as a probability with uncertainty for printing in a latex table.
    
    x: np.ndarray   
        Distribution data
    d: int=2
        Number of decimals to show
    r: List[int]=[5,95]
        Distribution interval to report as uncertainty.
    percent: bool=True
        Whether to convert to percent or state as a probability
    thresh: float=1e-5
        Threshold below which we will report a zero
    agg: Callable=np.mean
        Aggregator function to use for summarizing distribution.
    """
    x_p = np.percentile(x, r) * (100 if percent else 1)
    x_m = agg(x) * (100 if percent else 1)
    if x_m < thresh: x_m = 0
    if x_p[0] < thresh: x_p[0] = 0
    if x_p[1] < thresh: x_p[1] = 0
    
    a = f'{x_m:.{d}g}'
    if a == '1e+02': a=100
    b = f'{x_p[0]:.{d}g}'
    if b == '1e+02': b=100
    c = f'{x_p[1]:.{d}g}'
    if c == '1e+02': c=100
    
    return (f'{a} [{b}, {c}]')

latex_table_template ="""
\\documentclass{{article}}
    % General document formatting
    \\usepackage[margin=0.7in]{{geometry}}
    \\usepackage[parfill]{{parskip}}
    \\usepackage[utf8]{{inputenc}}
    \\usepackage{{pdflscape}}
    
    % Related to math
    \\usepackage{{amsmath,amssymb,amsfonts,amsthm}}
    \\usepackage{{booktabs}}

\\begin{{document}}
\\begin{{landscape}}
\\input{{{}}}
\\end{{landscape}}d
\\end{{document}}
"""

def calculate_rare_event_table(
        prob_rareevent: Dict,
        forecast_df: pd.DataFrame=data_MPSforecast,
        fat_thresholds: Dict[str, List[int]]=fat_thresholds,
        samp_years: np.ndarray=samp_years,
        ):
    """Generate table of forecasts for rare events.
    
    Parameters
    ----------
    prob_rareevent: Dict
        Nested dictionary of results from calc_rare_event_prob.
        Nesting levels should be fat_var, y_min, model, thresh
    forecast_df: pd.DataFrame=data_MPSforecast
        Population forecast dataframe that specifies scenarios and
        windows that projections are made for.
    fat_thresholds: Dict[str, List[int]]=fat_thresholds
        Projection thresholds used for each fat_var
    samp_years: np.ndarray=samp_years
        Array of years over which the projection is sampled.
    
    Returns
    -------
    prob_rareevent_table: Dict[str, Dict[str, List[pd.Series]]
        Nested dictionary.  Key levels are fat_vars, y_mins.
        Series are formatted table rows with indexes of scenario, window, model
        and columns are individual and cumulative event probabilities.
    prob_rareevent_cols: Dict[str, List[str]]
        Dictionary with keys of fat_vars and values that are lists of the columns
        to be found in prob_rareevent_table
    """
    ## Lookup config parameters from prob_rareevent dict
    fat_vars = list(prob_rareevent.keys())
    y_mins = list(prob_rareevent[fat_vars[0]].keys())
    ## Generate table
    prob_rareevent_table = {}
    prob_rareevent_cols = {}
    for fat_var in fat_vars:
        prob_rareevent_table[fat_var] = {}
        for y_min in y_mins:
            prob_rareevent_table[fat_var][y_min] = []
            for model in stan_models:
                for i_s, scenario in enumerate(forecast_df.index):
                    for i_w, window in enumerate(forecast_df.columns):
                        dic = {
                                'Scenario': scenario,
                                'Window': window,
                                'Model': model,
                                }
                        prob_rareevent_cols[fat_var] = [[],[]]
                        for thresh in sorted(fat_thresholds[fat_var]):
                            prob_rareevent_cols[fat_var][0] += ['$P_{'+str(samp_years[0])+'}(x>'+str(thresh)+')$']
                            prob_rareevent_cols[fat_var][1] += ['$P_{'+str(samp_years[0])+'-'+str(samp_years[-1])+'}(x>'+str(thresh)+')$']
                            ## Individual probability of first year
                            dic[prob_rareevent_cols[fat_var][0][-1]] = format_prob(
                                prob_rareevent[fat_var][y_min][model][thresh][1][:, i_s, i_w],
                                d=2)
                            ## Cumulative Probability of Last Year
                            dic[prob_rareevent_cols[fat_var][1][-1]] = format_prob(
                                prob_rareevent[fat_var][y_min][model][thresh][2][:, i_s, i_w, -1],
                                d=2)
                        prob_rareevent_table[fat_var][y_min] += [pd.Series(dic)]
    return prob_rareevent_table, prob_rareevent_cols

def draw_heatmap(*args, **kwargs):
    """Seaborn FacetGrid-compatible heatmap function that pulls x and y data from columns.
    
    From https://stackoverflow.com/questions/41471238/how-to-make-heatmap-square-in-seaborn-facetgrid
    """
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)

logit_label = r'$\rm{logit}[p(x_{\rm{min}}=10)] - \rm{logit}[p(x_{\rm{min}}=4)]$'
def plot_heatmaps_all_variables(data: pd.Series, 
                                fat_var: str,
                                value_label: str=logit_label):
    """Plot a FacetGrid of heatmaps, one for each model and window combination.
    The heatmaps show the value of the Series data along an x, y axis of
    x = threshold, y = window.
    
    Parameters
    ----------
    data: pd.Series
        Data series to plot.
        Should have index of ['variable', 'model', 'threshold', 'scenario', 'window'] 
    fat_var: str
        Fatality variable being plotted to extract from Series.
    value_label: str=logit_label
        Label for colorbar, corresponding to the meaning of the Series value.
    """
    df = data.loc[fat_var].reset_index()
    g = sns.FacetGrid(df, row='scenario', col='model', margin_titles=True)
    cbar_ax = g.fig.add_axes([.87, .3, .03, .4])
    g.map_dataframe(draw_heatmap, 'threshold', 'window', data.name,
                    square=True, 
                    vmin=-np.abs(data).max(), 
                    vmax=np.abs(data).max(),
                    cmap=plt.cm.RdBu,
                    cbar_ax=cbar_ax,
                    cbar_kws={'label': logit_label})
    g.fig.tight_layout(rect=[0, 0, .85, 1])
    plt.suptitle(f"Cumulative risk difference in log odds between tail locations for {fat_var}")

def plot_heatmaps_averaged(data: pd.Series, 
                                value_label: str=logit_label):
    """Plot a FacetGrid of heatmaps, one for each variable.
    The heatmaps show the value of the Series data along an x, y axis of
    x = threshold, y = model.  The scenario and window axes are averaged over.
    
    Parameters
    ----------
    data: pd.Series
        Data series to plot.
        Should have index of ['variable', 'model', 'threshold', 'scenario', 'window'],
        same as plot_heatmaps_all_variables
    value_label: str=logit_label
        Label for colorbar, corresponding to the meaning of the Series value.
    """
    df = data.groupby(['variable', 'threshold', 'model']).mean().reset_index()
    g = sns.FacetGrid(df, col='variable')
    cbar_ax = g.fig.add_axes([.87, .3, .03, .4])
    g.map_dataframe(draw_heatmap, 'threshold', 'model', data.name,
                    square=True, 
                    vmin=-np.abs(data).max(), 
                    vmax=np.abs(data).max(),
                    cmap=plt.cm.RdBu,
                    cbar_ax=cbar_ax,
                    cbar_kws={'label': logit_label})
    g.fig.tight_layout(rect=[0, 0, .85, 1])
    plt.suptitle(f"Cumulative risk difference in log odds between tail locations")
