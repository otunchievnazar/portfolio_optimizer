
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

assets = ['TSLA','AMZN','AAPL','NFLX', 'GOOG']
weights = np.array([0.2,0.2,0.2,0.2,0.2,])

stockStartDate = '2016-01-01'

today = datetime.today().strftime('%Y-%m-%d')

df = pd.DataFrame()

for stock in assets:
  df[stock] = web.DataReader(stock, data_source='yahoo', start = stockStartDate, end = today)['Adj Close']

title = 'Portfolio Adj. Close Price History'

my_stocks = df 

for c in my_stocks.columns.values:
  plt.plot(my_stocks[c],label = c)

plt.title(title)  
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Adj. Price USD ($)', fontsize = 18)
plt.legend(my_stocks.columns.values, loc= 'upper left')
plt.show()

returns = df.pct_change()

cov_matrix_annual = returns.cov() * 252

port_variance = np.dot( weights.T, np.dot(cov_matrix_annual, weights))


port_volatility = np.sqrt(port_variance)

portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights * 252)


percent_var = str(round(port_variance, 2) * 100) + '%'
percent_vols = str(round(port_volatility, 2) * 100) + '%'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + '%'

print('Expected annual return: '+ percent_ret)
print('Annual volatility / risk: '+ percent_vols)
print('Annual variance: '+ percent_var)



#Portfolio optimazation
from pypfopt.efficient_frontier import EfficientFrontier 
from pypfopt import risk_models
from pypfopt import expected_returns

# Calculate the expected returns and the annualised sample covariance matrix of a set returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#Optimize for max sharp ratio 
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose = True)

# get the allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)

da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = 100000)

allocation, leftover = da.lp_portfolio()
print('Discrete allocation:', allocation)
print('Funds remaining: ${:.2f}'.format(leftover))