
import numpy as np
import pandas as pd 
import statsmodels.api as sm
from pandas.tseries.offsets import BDay

def study_days_to_fomc_x_prediction_error_dependency(input): 

    # These are the separators for the buckets that are used to plot Image 3: 
    separators = [0,20,40,60,80,100,120,140,160,180,200]

    def create_categories(input, separators): 
        input["category"] = np.nan
        i = 1
        for x,y in zip(separators[1:], separators[0:-1]): 
            input["category"] = np.where((input["days_to_fomc"] < x) & (input["days_to_fomc"] >= y), i, input["category"] )
            i = i + 1 
        input["jump_soon"] = np.where(input["days_to_fomc"] <= 1, 1.0, 0.0)
        # category medians: 
        input["category"] = (input["category"] * 20) - 10 
        return input

    Q = input.copy()
    # The jump takes place on the next business days from FOMC meeting: 
    Q["jump-date"] = pd.to_datetime(Q["fomc_date"]) + BDay(1)
    Q["error"] = Q["effr_delta"] - Q["jump_sum"]
    Q = create_categories(Q, separators)

    QQ = Q.groupby('category')['error'].std().reset_index()    
    QQ['ones'] = 1.0 # For the regression line: 
    QQ_line = sm.OLS( QQ["error"].values, QQ[["ones", "category"]].values )
    QQ_result = QQ_line.fit()
    QQ_params = QQ_result.params
    QQ_rsq = QQ_result.rsquared
    regression_line_xs = [5, 195]
    regression_line_ys = [QQ_params[0] + 5 * QQ_params[1], QQ_params[0] + 195 * QQ_params[1]]

    ax0 = QQ.plot.scatter(x ='category', y ='error', )
    ax0.set_title('category median x error stdev') 
    ax0.plot(regression_line_xs, regression_line_ys, marker='o', color = 'orange')

    QQQ = Q.groupby('category')['effr_delta'].std().reset_index()
    QQQ['ones'] = 1.0
    QQQ_line = sm.OLS( QQQ["effr_delta"], QQQ[["ones", "category"]].values )
    QQQ_result = QQQ_line.fit()
    QQQ_params = QQQ_result.params
    QQQ_rsq = QQQ_result.rsquared
    regression_line_ys = [QQQ_params[0] + 5 * QQQ_params[1], QQQ_params[0] + 195 * QQQ_params[1]]

    ax1 = QQQ.plot.scatter(x ='category', y ='effr_delta')
    ax1.set_title('category median x delta(effr) stdev')
    ax1.plot(regression_line_xs, regression_line_ys, marker='o', color = 'orange')

    # For latex: 
    # [print((x,y)) for x,y in zip(QQ["category"].values, 100 * QQ["error"].values)]
    # line_to_plot = [QQ_params[0] + 5 * QQ_params[1], QQ_params[0] + 195 * QQ_params[1]]
    # line_to_plot = [100 * x for x in line_to_plot]
    # print('')
    # [print((x,y)) for x,y in zip(QQQ["category"].values, 100 * QQQ["effr_delta"].values)]
    # line_to_plot1 = [QQQ_params[0] + 5 * QQQ_params[1], QQQ_params[0] + 195 * QQQ_params[1]]
    # line_to_plot1 = [100 * x for x in line_to_plot1]
    return

def compute_delta_stdevs(parameters, deltas): 
    
    deltas_copy = deltas.copy()
    deltas_copy = deltas_copy.reset_index()
    fomc_column = 'alt_days_to_fomc'
    tenors = ["1mo", "3mo", "6mo"]
    parameters['date'] = pd.to_datetime(parameters['date'])
    deltas['date'] = pd.to_datetime(deltas['date'])
    X = parameters.merge(deltas_copy, on = 'date')
    X = X.loc[X['fomc'] == 0]
    all_results = pd.DataFrame()
    for x in tenors: 
        XX = X.loc[X['tenor'] == x]
        XX_short = XX.loc[XX[fomc_column] < 7]
        XX_medium_short = XX.loc[(XX[fomc_column] >= 7) & (XX[fomc_column] < 21)] 
        XX_medium_long = XX.loc[(XX[fomc_column] >= 21) & (XX[fomc_column] < 35)]
        XX_long = XX.loc[XX[fomc_column] >= 35]
        this_result = pd.DataFrame( { 
                                'tenor' : x,
                                'upcoming_jump': [XX_short['delta_5d'].std()], 
                                'medium_short': [XX_medium_short['delta_5d'].std()], 
                                'medium_long': [XX_medium_long['delta_5d'].std()],  
                                'long': [XX_long['delta_5d'].std()] })
        all_results = pd.concat([all_results, this_result])
    all_results = all_results.set_index('tenor')
    return all_results