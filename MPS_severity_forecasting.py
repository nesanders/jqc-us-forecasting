#!/usr/bin/env python3
from utils import *

#####################################
### Fit Stan models
#####################################

sm_dic = {}

## Compile models
for model in stan_models:
    sm_dic[model] = pystan.StanModel(file=stan_models[model][0], 
                                    includes=stan_models[model][1], 
                                    allow_undefined=True)

fit_dic = {}
ext_dic = {}
## Sample
for fat_var in fat_vars:
    fit_dic[fat_var] = {}
    ext_dic[fat_var] = {}
    
    for y_min in y_mins:
        
        fit_dic[fat_var][y_min] = {}
        ext_dic[fat_var][y_min] = {}
        
        # Format data for Stan
        sel = data[fat_var] >= y_min
        stan_data = {
            'N': len(data[sel]),
            'y': data[fat_var][sel].values.astype(int), 
            'y_min': y_min
            }

        ## Fit models
        for model in stan_models:
            fit_dic[fat_var][y_min][model] = sm_dic[model].sampling(
                            data=stan_data, iter=Niter, chains=Nchains)
            ext_dic[fat_var][y_min][model] = fit_dic[fat_var][y_min][model].extract()


#####################################
### Diagnostic plots
#####################################

######### Traceplots
#########

for fat_var in fat_vars:
    for y_min in y_mins:
        for model in stan_models:
            traceplot(ext_dic[fat_var][y_min][model], title=model)
            plt.savefig(odir['plots'] + 'fig_traceplot_'+model+'_'+fat_var+'_'+str(y_min)+'.pdf', bbox_inches='tight')

######### Model comparison statistics
#########

loo_dic = {}
for fat_var in fat_vars:
    loo_dic[fat_var] = {}
    for y_min in y_mins:
        loo_dic[fat_var][y_min] = {}
        for model in stan_models:
            loo_dic[fat_var][y_min][model] = {}
            logging.info(f"Calculating ELPD for {fat_var}, {y_min}, {model}")
            ## Calculate total ELPD
            idata = az.from_pystan(fit_dic[fat_var][y_min][model], log_likelihood="log_likelihood")
            loo_result = az.loo(idata, scale='log', pointwise=True)
            loo_dic[fat_var][y_min][model]['total'] = loo_result.loo
            loo_dic[fat_var][y_min][model]['total_se'] = loo_result.loo_se
            ## Calculate ELPD over points for x>10
            idata_gte_10 = az.from_pystan(fit_dic[fat_var][y_min][model], log_likelihood="log_likelihood_gte_10")
            loo_result_gte_10 = az.loo(idata_gte_10, scale='log', pointwise=True)
            if y_min <10:
                assert loo_result_gte_10.loo > loo_result.loo
            else:
                assert loo_result_gte_10.loo == loo_result.loo
            loo_dic[fat_var][y_min][model]['>10'] = loo_result_gte_10.loo
            loo_dic[fat_var][y_min][model]['>10_se'] = loo_result_gte_10.loo_se

## Plot LOO
for elpd_type in ['total', '>10']:
    loo_plot(loo_dic, elpd_type, sharey='col' if elpd_type=='total' else 'row')
    plt.savefig(odir['plots'] + f'fig_model_LOO_compare_{elpd_type}.pdf', bbox_inches='tight')

## Plot ELPD for each x_min
compare_elpd = np.zeros([len(y_mins), len(fat_vars), len(stan_models), len(stan_models)])
for i_ymin, y_min in enumerate(y_mins):
    for elpd_type in ['total', '>10']:
        compare_elpd[i_ymin] = loo_plot_per_x_min(loo_dic, y_min, elpd_type)
        plt.savefig(odir['plots'] + f'fig_model_LOO_compare_ymin_{y_min}_{elpd_type}.pdf',
                    bbox_inches='tight')

## Plot point-wise likelihoods
loo_plot_pointwise_likelihood(fit_dic, ext_dic)
plt.savefig(odir['plots'] + 'fig_likelihood_compare.pdf', bbox_inches='tight')

plt.close('all')


#####################################
### Output plots
#####################################

######### Probability density
#########

def plot_prob_density(fat_var, y_min, model, ccdf):
    individual_dist_fit(ext_dic[fat_var][y_min][model], scipy_models[model], model_name=model, 
                actual_data = data[fat_var].values.astype(int),
                inset = 'sigma' if model == 'lognormal' else 'alpha', xlab=fat_var,
                y_min=y_min, ccdf=ccdf, 
                hbins=1000 if ccdf else 30)
    plt.savefig(odir['plots'] + f'fig_dist_{model}_{fat_var}_{y_min}{("_ccdf" if ccdf else "")}.pdf', 
                bbox_inches='tight')

Parallel(n_jobs=njobs)(delayed(plot_prob_density)(fat_var, y_min, model, ccdf) 
                       for fat_var, y_min, model, ccdf
                       in itertools.product(fat_vars, y_mins, stan_models, [True, False]))

plt.close('all')

######### All-model comparison for each y_min
#########

def all_model_compare_within_y_min(y_min, ccdf):
    fig, axs = plt.subplots(1, len(fat_vars), figsize=(8,5), sharex='all', sharey='all')
    for i_var, fat_var in enumerate(fat_vars):
        all_dist_fit(ext_dic[fat_var][y_min], scipy_models, 
                    data[fat_var].values.astype(int), ax=axs[i_var], 
                    xlab=fat_var, y_min=y_min, ccdf=ccdf,
                    hbins=1000 if ccdf else 30,
                    ylim=[1e-5, 1])
        axs[i_var].tick_params(top=True, right=True)

    for i in range(1, len(fat_vars)): 
        axs[i].set_ylabel(None)
        axs[i].get_legend().remove()
    plt.savefig(odir['plots'] + f'fig_dist_allcompare_ymin{y_min}{"_ccdf" if ccdf else ""}.pdf', 
                bbox_inches='tight')

Parallel(n_jobs=njobs)(delayed(all_model_compare_within_y_min)(y_min, ccdf) 
                       for y_min, ccdf
                       in itertools.product(y_mins, [True, False]))

plt.close('all')


######### All-model comparison across y_mins
#########

def all_model_compare_across_y_min(fat_var, ccdf):
    fig, axs = plt.subplots(1, len(y_mins), figsize=(8,5), sharey='all')
    for i_ymin, y_min in enumerate(y_mins):
        all_dist_fit(ext_dic[fat_var][y_min], scipy_models, 
                    data[fat_var].values.astype(int), ax=axs[i_ymin], 
                    xlab='$x_{{\\rm{{min}}}}={}$'.format(y_min), y_min=y_min,
                    ccdf=ccdf, hbins=1000 if ccdf else 30,
                    ylim=[1e-5, 1] if ccdf else [1e-7,1])
        axs[i_ymin].tick_params(top=True, right=True)
    
    axs[0].legend(loc='upper right')
    for i in range(1, len(y_mins)): 
        axs[i].set_ylabel(None)
        axs[i].get_legend().remove()
    plt.suptitle(fat_var)
    plt.savefig(odir['plots'] + f'fig_dist_allcompare_{fat_var}{"_ccdf" if ccdf else ""}.pdf', 
                bbox_inches='tight')

Parallel(n_jobs=njobs)(delayed(all_model_compare_across_y_min)(fat_var, ccdf) 
                       for fat_var, ccdf
                       in itertools.product(fat_vars, [True, False]))

plt.close('all')


######### Probability of rare events (analytic)
#########

## Setup configurations
prob_rareevent = {}; all_configs = []
for fat_var in fat_vars:
    prob_rareevent[fat_var] = {}
    for y_min in y_mins:
        prob_rareevent[fat_var][y_min] = {}
        for model in stan_models:
            prob_rareevent[fat_var][y_min][model] = {}
            for thresh in fat_thresholds[fat_var]:
                all_configs += [[fat_var, y_min, model, thresh]]

## Calculate probabilities in parallel
calc_rare_event_prob_output = Parallel(n_jobs=njobs)(delayed(calc_rare_event_prob)(
                                    ext_dic[fat_var][y_min][model], scipy_models[model], 
                                    target_fatal=thresh, y_min=y_min, fat_var=fat_var) 
                       for fat_var, y_min, model, thresh in all_configs)

## Put results in the prob_rareevent dictionary
for i, (fat_var, y_min, model, thresh) in enumerate(all_configs): 
    prob_rareevent[fat_var][y_min][model][thresh] = calc_rare_event_prob_output[i]
del calc_rare_event_prob_output

## Plot                
for fat_var in fat_vars:
    for y_min in y_mins:
        for model in stan_models:
            for thresh in fat_thresholds[fat_var]:
                plot_rare_event_lines(prob_rareevent[fat_var][y_min][model][thresh][2],
                                    target_fatal=thresh, y_min=y_min, fat_var=fat_var,
                                    model_name=model.title())
                plt.savefig(odir['plots'] + 'fig_rare_event_analytic_'+model+'_'+fat_var+'_'+str(thresh)+'_'+str(y_min)+'.pdf', bbox_inches='tight')
                plt.close()

dump(prob_rareevent, odir['data'] + 'prob_rareevent.joblib.bz2', compress=3)


######### Probability of rare events comparisons
#########

## Load means into a DataFrame
cumulative_rare_event_df = pd.DataFrame(
    columns = {'variable', 'y_min', 'model', 'threshold', 'scenario', 'window', 'cumulative probability'})
for fat_var in fat_vars:
    for y_min in y_mins:
        for model in stan_models:
            for thresh in fat_thresholds[fat_var]:
                for i_scenario, scenario in enumerate(data_MPSforecast.index):
                    for i_window, window in enumerate(data_MPSforecast.columns):
                        ## Calculate mean probability at last year
                        _ = prob_rareevent[fat_var][y_min][model][thresh][2][:, i_scenario, i_window, -1]
                        cumulative_rare_event_df = cumulative_rare_event_df.append({
                            'variable': fat_var, 
                            'y_min': y_min, 
                            'model': model, 
                            'threshold': thresh, 
                            'scenario': scenario, 
                            'window': window,
                            'cumulative probability': np.mean(_)}, 
                        ignore_index=True)

## Logit transform
cumulative_rare_event_df['logit cumulative probability'] = logit(cumulative_rare_event_df['cumulative probability'])
## Set index
cumulative_rare_event_df = cumulative_rare_event_df.set_index(['y_min', 'variable', 'model', 'threshold', 'scenario', 'window'])
## Calculate difference between y_mins
cumulative_diff_rare_event_df = (cumulative_rare_event_df.loc[10] - cumulative_rare_event_df.loc[4])['logit cumulative probability']

## Make plot grid for each variable
for fat_var in fat_vars:
    plot_heatmaps_all_variables(cumulative_diff_rare_event_df, fat_var)
    plt.savefig(odir['plots'] + f'fig_log_odds_diff_{fat_var}.pdf', bbox_inches='tight')

## Since the results don't vary much by scenario and window, we can also just average over those variables to simplify the plots
plot_heatmaps_averaged(cumulative_diff_rare_event_df)
plt.savefig(odir['plots'] + f'fig_log_odds_diff_average.pdf', bbox_inches='tight')


######### Probability of rare events (analytic) Table
#########

prob_rareevent_table, prob_rareevent_cols = calculate_rare_event_table(prob_rareevent)

for fat_var in fat_vars:
    for y_min in y_mins:
        temp_df = pd.concat(prob_rareevent_table[fat_var][y_min], axis=1, sort=True).T
        ## Convert window to integer so it appears in the right order
        temp_df['Window'] = temp_df['Window'].apply(lambda x: int(x[:-2]))
        colorder = {
            'SingleYear':prob_rareevent_cols[fat_var][0], 
            'Cumulative':prob_rareevent_cols[fat_var][1]
            }
        temp_df = temp_df.set_index(['Scenario','Model','Window']).sort_index()
        for co in colorder:
            fname = 'tab_scen_{}_{}_{}.tex'.format(fat_var.replace(' ', '_'), co, y_min)
            # Save a csv output in case needed
            temp_df.loc[:,colorder[co]].to_csv(fname+'.csv')
            # Save a tex output and compile to pdf
            temp_df.loc[:,colorder[co]].to_latex(fname, escape=False)
            wname = odir['tables'] + 'wrap_'+fname
            with open(wname, 'w') as f:
                f.write(latex_table_template.format(fname))
            os.system(f'pdflatex "{wname}"')
            os.system('rm "{}"*.aux'.format('wrap_'+fname.split('.tex')[0]))
            os.system('rm "{}"*.log'.format('wrap_'+fname.split('.tex')[0]))
            #os.system('rm "{}"'.format(fname)) # Do not delete source tex file
            os.system('mv "{}"*.pdf {}'.format('wrap_'+fname.split('.tex')[0], odir['tables']))
