---
layout: wide_default
---   

# Abstract

For this analysis, I looked at the list of companies that were listed on the S&P 500 list as of March 1st, 2020.
In addition to the returns they generated when the pandemic hit (week of March 9th-March 13th), I also took into account different accounting measures to evaluate whether some of these companies exhibited specific risk factors that led them to be more vulnerable to the pandemic. These risk factors were measured using Natural Language Processing (NLP), specifically the Near Regex package. The majority of this analysis centered around correlations and linear regressions. 

# Analysis - What are some risk factors that made companies more vulnerable to the COVID-19 pandemic?

First, the file that includes returns as well as accounting data will be loaded.


```python
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from datetime import datetime
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols as sm_ols
import matplotlib.pyplot as plt

#add returns file first

#create url
url = ('https://github.com/LeDataSciFi/ledatascifi-2022/blob/main/data/2019-2020-stock_rets%20cleaned.zip?raw=true')

#open it
with urlopen(url) as request:
    data = BytesIO(request.read())

with ZipFile(data) as archive:
    with archive.open(archive.namelist()[0]) as stata:
        stock_returns = pd.read_stata(stata)
        
#get dataset with just the dates we need

start_date = datetime(2020, 3, 9) #taken from https://ledatascifi.github.io/ledatascifi-2022/content/05/05b_capm.html?highlight=datetime
end_date = datetime(2020, 3, 13)

#convert stock returns date to date-time format
#taken from https://sparkbyexamples.com/pandas/pandas-convert-integer-to-datetime/#:~:text=Use%20DataFrame.,convert%20integer%20to%20datetime%20formate.
stock_returns['date'] = pd.to_datetime(stock_returns['date'].astype(str))

week = stock_returns.query('date >= @start_date & date <= @end_date')

week['ret'] = pd.to_numeric(week['ret'],errors='coerce') #make sure return variable is formatted as an int so that it can be used for analysis

#get return for week of Mar 9 - Mar 13, 2020 (the cumulative return for the week)

week_returns = (week
   # compute gross returns for each asset
   .assign(R = 1+week['ret'])
   # for each portfolio and time period...
   .groupby(['ticker'])
   # get the gross returns, and cumulate by taking the product
   ['R'].prod()
   # subtract one to get back to simple returns
   -1
)
week_returns = week_returns.to_frame()

#read file in output folder and merge with weekly return
firms_df = pd.read_csv('output/sp500_accting_plus_textrisks.csv')
firms_df = firms_df.merge(week_returns, how='left', left_on='Symbol', right_on='ticker', indicator = 'exists', validate='one_to_one')
firms_df.pop('Unnamed: 0')
```

    /var/folders/dt/qsms415d59sdwybzh19f_s7m0000gn/T/ipykernel_57707/2726115983.py:36: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      week['ret'] = pd.to_numeric(week['ret'],errors='coerce') #make sure return variable is formatted as an int so that it can be used for analysis





    0        0
    1        1
    2        2
    3        3
    4        4
          ... 
    500    500
    501    501
    502    502
    503    503
    504    504
    Name: Unnamed: 0, Length: 505, dtype: int64



# Risk Measurements

There are three distinct risk factors that were evaluated, with one of them being modified twice (Supply Chain risk), getting us up to 5 variables available for analysis. These are:
- Supply Chain Risk (with two variations)
- Reputation Risk
- Labor Risk

Let's look at them one by one.

## Supply Chain Risk

With companies like [Deloitte calling out supply chain disruptions](https://www2.deloitte.com/global/en/pages/risk/cyber-strategic-risk/articles/covid-19-managing-supply-chain-risk-and-disruption.html) as a vulnerable factor to firms during COVID-19, it leads one to believe that companies that have a more robust supply chain, or perhaps one that does not heavily rely on Asia/China, more resilient to the pandemic. As international travel came to a hault almost overnight in March 2020, supply chains that operate on a broad scale, and especially those that depend on the region that is the source of the pandemic, presumably left companies struggling to fulfill orders and maintain their international presence. As such, I wanted to measure to what extent companies that identified vulnerabilities in their supply chain were affected by the pandemic (or at least at its very beginning). 

For the **first measure**, two sets of words were evaluated: 
- supply chain, supplies, raw material, raw materials, manufacturing, manufacture, products, production
- risk, susceptible, international, foreign, issue, problem

I looked at how close these words were to each other within a 50 word limit as exploring 10-k's manually led me to believe that looking within a single sentence might be too narrow of a search, while looking at significantly more words would make my search too broad. I focused on raw materials, production, manufacturing, and supply chain in my search as it relates to foreign production and risk. 

The **second measure** used the exact same words but narrowed the search to 10 words (approximately one sentence) that yielded far less hits (explained in more detailed later). 

The **third measure** slighlty modified the words used to focus more on imports and exports coming in/out of China and Asia as this is a particular region of concern. The sets of words are:
- import, imports, export, exports, source, sourcing, supply chain, suppliers, inventory, supplies, china, chinese, asia, asian
- concern, issue, dependent, dependency, constraints, vendor, vendors


**Basic statistics surrounding this measurement are included below:**


```python
firms_df[['Supply_Chain_Risk_1','Supply_Chain_Risk_2', 'Supply_Chain_Risk_3']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Supply_Chain_Risk_1</th>
      <th>Supply_Chain_Risk_2</th>
      <th>Supply_Chain_Risk_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>489.000000</td>
      <td>489.000000</td>
      <td>489.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>17.511247</td>
      <td>6.163599</td>
      <td>2.613497</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.215541</td>
      <td>6.794542</td>
      <td>3.319697</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>15.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>26.000000</td>
      <td>9.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>128.000000</td>
      <td>65.000000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>



As we can see, there are 489 of all of these measurements for our firms as I was able to download 489 10-k reports. We can see that the first measurement yielded the highest average number of hits (with the most deviation from the mean, however), while the third had the lowest number of hits. This makes sense as the first one was much more broad and included hits within a 50 word span (as opposed to the second measurement), while the third one focused more on Asia and China, and less results were expected. It is important to note that means for the first two measurements are slightly skewed due to outliers (note max values in the table above).

**Supply chain risk measurements exhibited quite a few high correlations with accounting measurements:**


```python
sc_corr = firms_df[[
       'Supply_Chain_Risk_1', 'Supply_Chain_Risk_2', 'Supply_Chain_Risk_3',
       'ppe_a', 'xrd_a', 'dltt_a'
       ]].corr()[:3]
sns.heatmap(sc_corr)
plt.show()
```


    
![png](output_13_0.png)
    


Based on the above, we can note a few high correlations:
- Supply_Chain_Risk_1(/2) and ppe_a: the higher the supply chain risk, the lower investment in PP&E. This can indicate that companies that have less efficient supply chains tend to not have as many assets. This can be interpreted as those companies not having as many fixed assets, resulting in less robust supply chains.
- Supply_Chain_Risk_1(/2) and xrd_a: the higher the supply chain risk, the higher the investment in R&D. A lot of companies that do not have a well developed supply chain might focus on selling services rather than products - especially technology, resulting in higher levels of innovation. 
- Supply_Chain_Risk_1(/2) and dltt_a: the higher the supply chain risk, the lower overall debt levels. More liquidity can signify a company highly focused on efficient operations, and those would need a highly efficient supply chain. 

No significant correlations were found with the third measurement. This can indicate that this measurement was a bit too narrow. It is also important to note that the interpretations of the above are hypotheses, not facts.

## Labor Risk

For the second risk measure, I focused on another resource that might make firms vulnerable if international mobility is compromised - international labor. As the [International Labour Organization suggests](https://news.un.org/en/story/2022/01/1109832), millions of people across the world have been forced to look for new forms of employment due to a sudden hault in tourist and other traveling related activities. It makes sense then to assume that a company that depends more on foreign workers might prove to be more vulnerable to pandemics as they might lose a significant portion of their workforce due to travel and immigration regulations. 

I again focused on words that are in the 50 word span of each other to not make my search too narrow nor too broad. The words that were used for this search were:
- labor, workers, employees, immigrants
- risk, susceptible, intensive, union, unions, foreign, international, issue, issues

**Basic statistics surrounding this measurement are included below:**


```python
firms_df['Labor_Risk'].describe().to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Labor_Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>489.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.012270</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.090558</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>46.000000</td>
    </tr>
  </tbody>
</table>
</div>



We again have 489 observations with a mean of 7 hits, similar to the second supply chain measurement. The standard deviation however is almost as large, indicating that there are quite a few outliers (max of 46). 

**Labor Risk had a high correlation with a few accounting measurements:**


```python
labor_corr = firms_df[['Labor_Risk', 'l_emp',
       'td_mv',
       'xrd_a'
       ]].corr()[:1]
sns.heatmap(labor_corr)
plt.show()
```


    
![png](output_21_0.png)
    


The above are the only notable correlations. Here are hypothetical explanations of them:
- Labor_Risk and l_emp: l_emp refers to log(employees) so we will obviously get a notable correlation there as the labor search includes the words employees and workers in it. 
- Labor_Risk and td_mv: the higher the labor risk, the higher the debt to equity ratio is. Firms with a big international presence might need more funding to fuel that growth, yielding a higher D/E ratio.
- Labor_Risk and xrd_a: the higher the labor risk, the lower the R&D ratio. R&D depends heavily on personnel and human resources and we would expect a company with low labor risk to be able to produce more innovation.

## Reputation Risk

For the third and final risk measure, I looked at reputation risk. By reputation, we're looking at how internal and external stakeholders view the company, and most importantly, whether they look at the company in a favorable view. Companies that garner more support are intuitively likely to be less vulnerable to a major event like a pandemic as their stakeholders, and especially investors, trust the company and its leadership more. This is exactly what [Cutting Edge PR talked about in their evaluation of over 5,000 reports](https://cuttingedgepr.com/good-corporate-reputation-is-vital-especially-during-covid-19-times/) on how reputation affects resiliency. 

Once again, I focused on words that are in the 50 word span of each other. The words that were used for this search were:
- reputation, prestige, reputable, prestigious, image, public image, notoriety, standing, prominence
- risk, susceptible, issue, concern, danger, problem, undermined, risks

**Basic statistics surrounding this measurement are included below:**


```python
firms_df['Reputation_Risk'].describe()
```




    count    489.000000
    mean       2.390593
    std        2.763342
    min        0.000000
    25%        1.000000
    50%        2.000000
    75%        3.000000
    max       29.000000
    Name: Reputation_Risk, dtype: float64



We again have 489 observations with a mean of 2 hits, similar to the third supply chain measurement. The standard deviation however is quite large, with the max observation being 29 and min being 0. The average low number of hits could indicate that my search was too narrow and/or that reputation is not a huge issue for a lot of firms in the S&P 500. This would make sense as these firms are some of the biggest globally, and probably have a decent reputation.

**Reputation Risk had a high correlation with a few accounting measurements:**


```python
rep_corr = firms_df[['Reputation_Risk',
       'Inv',
       'mb','ppe_a'
       ]].corr()[:1]
sns.heatmap(rep_corr)
plt.show()
```


    
![png](output_29_0.png)
    


The above are the only notable correlations. Here are hypothetical explanations of them:
- Reputation_Risk and Inv: the higher the reputation risk, the less inventory a firm has (a consequently maybe the lower level of sales it has too). If a firm has a challenging reputation, they might sell less.
- Reputation_Risk and mb: the higher the reputation risk, the higher the market to book ratio is. A high market to book ratio usually indicates that a stock is expensive, so this correlation is somewhat counter-intuitive. It could point to potential problems with the risk measurement itself.
- Reputation_Risk and ppe_a: the higher the reputation risk, the lower PP&E. This could indicate that companies with a challenging reputation might not be as big in size and thus would not need as much PP&E.

## Validation Checks - are these risk measurements likely to yield authentic and valid results?

Some of these measurements are more likely to be valid than others. For instance, **labor risk** might yield a few hits that focus on labor unions rather than international workers/issues. This can be seen here in Cintas' 10-k: 

"In total, Cintas has approximately 11,400 local delivery routes, 470 operational facilities and 11 distribution centers. At May 31, 2019, Cintas employed approximately 45,000 employee-partners, of which approximately 1,600 were represented by **labor unions.**"

We can see that the above does not necessarily discuss international risk as it pertains to foreign workers, but in turn discusses labor unions. Thus, this risk might not be as reliable as other ones.

However, with **supply chain risks**, I specifically focused on words relating to risk and supply chain, manufacturing, production etc. that might yield more targeted hits. While manually validating 10-ks it was far easier to find hits for this risk. An example of a hit is diplayed below (taken from ABIOMED Inc.):

"Some of our suppliers may be the only source for a particular component, which makes us vulnerable to significant cost increases or shortage of supply. We have many **foreign** suppliers for some of our parts in which we are subject to currency exchange rate volatility. Some of our vendors are small in size and may have difficulty supplying the quantity and quality of materials required for our **products** as our business grows. Vendors that are the sole source of certain products may decide to limit or eliminate sales of certain components due to product liability or other concerns and we might not be able to find a suitable replacement for those products."

However, the exact opposite is true for the third supply chain risk measurement that captured China and Asia and how they relate to supply chain risk. This, again, points to the fact that this is a very targeted search that a lot of these companies likely would not mention in their 10-ks. Still, big companies such as Nike, Inc. that depend a lot on China for their operations yielded hits such as:

"Depending on the extent that certain new or proposed reforms are implemented by the U.S. government and the manner in which foreign governments respond to such reforms, it may become necessary for us to change the way we conduct business, which may adversely affect our results of operations. In addition, with respect to proposed trade restrictions targeting **China**, which represents an important sourcing country and consumer market for us, we are working with a broad coalition of global businesses and trade associations representing a wide variety of sectors to help ensure that any legislation enacted and implemented (i) addresses legitimate and core **concerns**, (ii) is consistent with international trade rules and (iii) reflects and considers China's domestic economy and the important role it has in the global economic community."

Finally, with **reputation risks**, I expect that this search would yield accurate results as the first set of words all focus on reputation and public image while the second is focused on issues, risks, and problems. These is not a lot of "wiggle room" here and room for interpretation. Still, as mentioned above, since we are looking at companies that are quite successful, they might not have extensive issues related to reputation, bringing about a low number of hits. Here is an example of a hit from the 10-k of Twitter, Inc.:

"We will also continue to experience media, legislative or regulatory scrutiny of our decisions regarding privacy, data protection, security, content and other **issues**, which may adversely affect our **reputation** and brand."

### What industries do the firms with the most hits (among each risk factor) belong to? 

**Supply_Chain_Risk_1:**


```python
firms_df[['Supply_Chain_Risk_1','sic']].query('Supply_Chain_Risk_1 > 55').sort_values(by = 'Supply_Chain_Risk_1', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Supply_Chain_Risk_1</th>
      <th>sic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>386</th>
      <td>128.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>83.0</td>
      <td>2834.0</td>
    </tr>
    <tr>
      <th>317</th>
      <td>73.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>72.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>297</th>
      <td>65.0</td>
      <td>3760.0</td>
    </tr>
    <tr>
      <th>504</th>
      <td>64.0</td>
      <td>2834.0</td>
    </tr>
    <tr>
      <th>397</th>
      <td>59.0</td>
      <td>3812.0</td>
    </tr>
    <tr>
      <th>106</th>
      <td>57.0</td>
      <td>2840.0</td>
    </tr>
    <tr>
      <th>259</th>
      <td>57.0</td>
      <td>3845.0</td>
    </tr>
    <tr>
      <th>236</th>
      <td>56.0</td>
      <td>3844.0</td>
    </tr>
  </tbody>
</table>
</div>



The industries in question are: Pharmaceutical Preparations; X-Ray Apparatus and Tubes and Related Irradiation Apparatus; Electromedical and Electrotherapeutic Apparatus; Search, Detection, Navigation, Guidance, Aeronautical, and Nautical Systems and Instruments; Soap, Detergents, and Cleaning Preparations, and Guided Missiles and Space Vehicles and Parts. 

All of these make a lot of sense as these industries contain highly sensitive equipment (especially medical, military, and pharmaceutical) that have to be especially cared for in transport. These also tend to have a lot of international firms as countries across the world require the same/very similar products in these industries. Thus, our search here didn't yield great surprises.

**Supply_Chain_Risk_2:**


```python
firms_df[['Supply_Chain_Risk_2','sic']].query('Supply_Chain_Risk_2 > 23.5').sort_values(by = 'Supply_Chain_Risk_2', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Supply_Chain_Risk_2</th>
      <th>sic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>386</th>
      <td>65.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>317</th>
      <td>35.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>32.0</td>
      <td>2834.0</td>
    </tr>
    <tr>
      <th>106</th>
      <td>30.0</td>
      <td>2840.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29.0</td>
      <td>3841.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>29.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>259</th>
      <td>29.0</td>
      <td>3845.0</td>
    </tr>
    <tr>
      <th>105</th>
      <td>28.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>236</th>
      <td>28.0</td>
      <td>3844.0</td>
    </tr>
    <tr>
      <th>327</th>
      <td>24.0</td>
      <td>2086.0</td>
    </tr>
    <tr>
      <th>397</th>
      <td>24.0</td>
      <td>3812.0</td>
    </tr>
  </tbody>
</table>
</div>



A lot of these industries are the same ones we saw above. Specifically, we have the following industries: Surgical and Medical Instruments, Pharmaceutical Preparation Manufacturers, Soap, Detergents, and Cleaning Preparations, Irradiation Apparatus Manufacturing, Electromedical and Electrotherapeutic Apparatus, Bottled and Canned Soft Drinks and Carbonated Waters, and Search, Detection, Navigation, Guidance, Aeronautical, and Nautical Systems and Instruments.  

This makes sense as the same words were included in the search, it's just the word span that changed. One thing that is important to note here is that while these results make sense, especially when looking at pharmaceutical companies, we know that they did extremely well during the pandemic, while this model presents them as those with a high supply chain risk (which they do have, however, this did not severely negatively impact them during the pandemic).

**Supply_Chain_Risk_3:**


```python
firms_df[['Supply_Chain_Risk_3','sic']].query('Supply_Chain_Risk_3 > 12').sort_values(by = 'Supply_Chain_Risk_3', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Supply_Chain_Risk_3</th>
      <th>sic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>94</th>
      <td>23.0</td>
      <td>5045.0</td>
    </tr>
    <tr>
      <th>70</th>
      <td>22.0</td>
      <td>5731.0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>21.0</td>
      <td>5211.0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>18.0</td>
      <td>5531.0</td>
    </tr>
    <tr>
      <th>447</th>
      <td>18.0</td>
      <td>5200.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>16.0</td>
      <td>3576.0</td>
    </tr>
    <tr>
      <th>280</th>
      <td>14.0</td>
      <td>5311.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13.0</td>
      <td>5531.0</td>
    </tr>
    <tr>
      <th>488</th>
      <td>13.0</td>
      <td>3572.0</td>
    </tr>
  </tbody>
</table>
</div>



In the third search, we have vastly different results than in the first two, which is quite interesting. Here, we have the following industries: Retail sale of new automobile tires, batteries, and other automobile parts and accessories;  Computer Communications Equipment;  Retail sale of radios, television sets, record players, stereo equipment, sound reproducing equipment, and other consumer audio and video electronics equipment (including automotive); Computers, computer peripheral equipment, and computer software; Retail stores; Lumber and Other Building Materials; Retail - Building Materials, Hardware, Garden Supply, and Computer Storage Devices.

Once again, these searches make sense. China is known for its production and exports in the retail, automotive, and computer markets, so we would expect supply chains that rely heavily on China and Asia to be in these industries. 

**Labor_Risk:**


```python
firms_df[['Labor_Risk','sic']].query('Labor_Risk > 18').sort_values(by = 'Labor_Risk', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Labor_Risk</th>
      <th>sic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>449</th>
      <td>46.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>392</th>
      <td>34.0</td>
      <td>1731.0</td>
    </tr>
    <tr>
      <th>190</th>
      <td>27.0</td>
      <td>4513.0</td>
    </tr>
    <tr>
      <th>225</th>
      <td>27.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>24.0</td>
      <td>4512.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>24.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>339</th>
      <td>24.0</td>
      <td>1040.0</td>
    </tr>
    <tr>
      <th>461</th>
      <td>21.0</td>
      <td>4210.0</td>
    </tr>
    <tr>
      <th>478</th>
      <td>21.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>270</th>
      <td>19.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



These are the industries that are represented in the top hits: Air Transportation; Furnishing air delivery of individually addressed letters, parcels, and packages, except by the U.S. Postal Service; Gold and Silver Ores; Electrical Contractors and Other Wiring Installation Contractors, and Trucking and Courier Services.

Once again, this search is promising as all of these industries require a lot of foreign labor - especially package delivery and air transportation. Moreover, a lot of electrical contractors and truck driving companies [hire immigrants](https://www.forbes.com/sites/andyjsemotiuk/2021/11/16/could-us-immigration-help-solve-need-to-recruit-1-million-new-truck-drivers-over-next-decade/?sh=4ec281ff66ba), again pointing to foreign labor. This is very promising as above we discussed that this search might catch a lot of false hits, especially because of labor unions. We can see here that our results might make more sense than we originally thought.

**Reputation_Risk:**


```python
firms_df[['Reputation_Risk','sic']].query('Reputation_Risk > 9').sort_values(by = 'Reputation_Risk', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reputation_Risk</th>
      <th>sic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>270</th>
      <td>29.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>192</th>
      <td>21.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>329</th>
      <td>15.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>86</th>
      <td>13.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>427</th>
      <td>13.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>219</th>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>148</th>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>141</th>
      <td>10.0</td>
      <td>8090.0</td>
    </tr>
    <tr>
      <th>317</th>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>326</th>
      <td>10.0</td>
      <td>2052.0</td>
    </tr>
    <tr>
      <th>334</th>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>344</th>
      <td>10.0</td>
      <td>3021.0</td>
    </tr>
  </tbody>
</table>
</div>



A lot of our top hits have unidentified industries. However, the ones we can identify are: Services - Misc Health & Allied Services; Cookies and Crackers, and Rubber and Plastics Footwear. 

This is quite interesting of a find that cannot be easily interpreted. Health and Allied Services will definitely depend a lot on reputation, and any company in any food-related industry will need to have a great reputation as their products can otherwise potentially harm their consumers' health. Still, there is not a clear pattern here, indicating potential issues with our search.

### Finally, what % of 0's does every risk contain?

**Supply_Chain_Risk_1:**


```python
100 - (
        (firms_df['Supply_Chain_Risk_1'] > 0)      
       .sum(axis=0)     
        /len(firms_df['Supply_Chain_Risk_1'])
        *100        
    ) 

```




    7.920792079207914



**Supply_Chain_Risk_2:**


```python
100 - (
        (firms_df['Supply_Chain_Risk_2'] > 0)      
       .sum(axis=0)     
        /len(firms_df['Supply_Chain_Risk_2'])       
        *100            
    ) 
```




    19.405940594059402



**Supply_Chain_Risk_3:**


```python
100 - (
        (firms_df['Supply_Chain_Risk_3'] > 0)     
       .sum(axis=0)     
        /len(firms_df['Supply_Chain_Risk_3'])       
        *100            
    ) 
```




    28.51485148514851



**Labor_Risk:**


```python
100 - (
        (firms_df['Labor_Risk'] > 0)      
       .sum(axis=0)     
        /len(firms_df['Labor_Risk'])       
        *100            
    ) 
```




    5.5445544554455495



**Reputation_Risk:**


```python
100 - (
        (firms_df['Reputation_Risk'] > 0)      
       .sum(axis=0)     
        /len(firms_df['Reputation_Risk'])       
        *100            
    ) 
```




    22.17821782178217



As expected, the biggest % of 0's are found in Supply_Chain_Risk_3 (due to the reasons discussed above), as well as Reputation_Risk, which makes sense given our pool of highly successful companies.

## Final Sample

Now that we've discussed all added risk measures as well as their potential validity, we can look at our dataset as a whole and do some exploratory analysis.


```python
print("The shape is: ",firms_df.shape, '\n---')
firms_df.describe().T.style.format('{:,.2f}')
```

    The shape is:  (505, 56) 
    ---





<style type="text/css">
</style>
<table id="T_72484_">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >count</th>
      <th class="col_heading level0 col1" >mean</th>
      <th class="col_heading level0 col2" >std</th>
      <th class="col_heading level0 col3" >min</th>
      <th class="col_heading level0 col4" >25%</th>
      <th class="col_heading level0 col5" >50%</th>
      <th class="col_heading level0 col6" >75%</th>
      <th class="col_heading level0 col7" >max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_72484_level0_row0" class="row_heading level0 row0" >CIK</th>
      <td id="T_72484_row0_col0" class="data row0 col0" >505.00</td>
      <td id="T_72484_row0_col1" class="data row0 col1" >754,679.65</td>
      <td id="T_72484_row0_col2" class="data row0 col2" >538,187.48</td>
      <td id="T_72484_row0_col3" class="data row0 col3" >1,800.00</td>
      <td id="T_72484_row0_col4" class="data row0 col4" >92,380.00</td>
      <td id="T_72484_row0_col5" class="data row0 col5" >874,761.00</td>
      <td id="T_72484_row0_col6" class="data row0 col6" >1,113,169.00</td>
      <td id="T_72484_row0_col7" class="data row0 col7" >1,757,898.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row1" class="row_heading level0 row1" >Supply_Chain_Risk_1</th>
      <td id="T_72484_row1_col0" class="data row1 col0" >489.00</td>
      <td id="T_72484_row1_col1" class="data row1 col1" >17.51</td>
      <td id="T_72484_row1_col2" class="data row1 col2" >15.22</td>
      <td id="T_72484_row1_col3" class="data row1 col3" >0.00</td>
      <td id="T_72484_row1_col4" class="data row1 col4" >5.00</td>
      <td id="T_72484_row1_col5" class="data row1 col5" >15.00</td>
      <td id="T_72484_row1_col6" class="data row1 col6" >26.00</td>
      <td id="T_72484_row1_col7" class="data row1 col7" >128.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row2" class="row_heading level0 row2" >Supply_Chain_Risk_2</th>
      <td id="T_72484_row2_col0" class="data row2 col0" >489.00</td>
      <td id="T_72484_row2_col1" class="data row2 col1" >6.16</td>
      <td id="T_72484_row2_col2" class="data row2 col2" >6.79</td>
      <td id="T_72484_row2_col3" class="data row2 col3" >0.00</td>
      <td id="T_72484_row2_col4" class="data row2 col4" >1.00</td>
      <td id="T_72484_row2_col5" class="data row2 col5" >4.00</td>
      <td id="T_72484_row2_col6" class="data row2 col6" >9.00</td>
      <td id="T_72484_row2_col7" class="data row2 col7" >65.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row3" class="row_heading level0 row3" >Supply_Chain_Risk_3</th>
      <td id="T_72484_row3_col0" class="data row3 col0" >489.00</td>
      <td id="T_72484_row3_col1" class="data row3 col1" >2.61</td>
      <td id="T_72484_row3_col2" class="data row3 col2" >3.32</td>
      <td id="T_72484_row3_col3" class="data row3 col3" >0.00</td>
      <td id="T_72484_row3_col4" class="data row3 col4" >0.00</td>
      <td id="T_72484_row3_col5" class="data row3 col5" >2.00</td>
      <td id="T_72484_row3_col6" class="data row3 col6" >3.00</td>
      <td id="T_72484_row3_col7" class="data row3 col7" >23.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row4" class="row_heading level0 row4" >Reputation_Risk</th>
      <td id="T_72484_row4_col0" class="data row4 col0" >489.00</td>
      <td id="T_72484_row4_col1" class="data row4 col1" >2.39</td>
      <td id="T_72484_row4_col2" class="data row4 col2" >2.76</td>
      <td id="T_72484_row4_col3" class="data row4 col3" >0.00</td>
      <td id="T_72484_row4_col4" class="data row4 col4" >1.00</td>
      <td id="T_72484_row4_col5" class="data row4 col5" >2.00</td>
      <td id="T_72484_row4_col6" class="data row4 col6" >3.00</td>
      <td id="T_72484_row4_col7" class="data row4 col7" >29.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row5" class="row_heading level0 row5" >Labor_Risk</th>
      <td id="T_72484_row5_col0" class="data row5 col0" >489.00</td>
      <td id="T_72484_row5_col1" class="data row5 col1" >7.01</td>
      <td id="T_72484_row5_col2" class="data row5 col2" >5.09</td>
      <td id="T_72484_row5_col3" class="data row5 col3" >0.00</td>
      <td id="T_72484_row5_col4" class="data row5 col4" >3.00</td>
      <td id="T_72484_row5_col5" class="data row5 col5" >6.00</td>
      <td id="T_72484_row5_col6" class="data row5 col6" >9.00</td>
      <td id="T_72484_row5_col7" class="data row5 col7" >46.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row6" class="row_heading level0 row6" >gvkey</th>
      <td id="T_72484_row6_col0" class="data row6 col0" >357.00</td>
      <td id="T_72484_row6_col1" class="data row6 col1" >43,204.34</td>
      <td id="T_72484_row6_col2" class="data row6 col2" >59,781.52</td>
      <td id="T_72484_row6_col3" class="data row6 col3" >1,045.00</td>
      <td id="T_72484_row6_col4" class="data row6 col4" >6,216.00</td>
      <td id="T_72484_row6_col5" class="data row6 col5" >12,540.00</td>
      <td id="T_72484_row6_col6" class="data row6 col6" >61,483.00</td>
      <td id="T_72484_row6_col7" class="data row6 col7" >316,056.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row7" class="row_heading level0 row7" >lpermno</th>
      <td id="T_72484_row7_col0" class="data row7 col0" >357.00</td>
      <td id="T_72484_row7_col1" class="data row7 col1" >53,567.94</td>
      <td id="T_72484_row7_col2" class="data row7 col2" >29,810.26</td>
      <td id="T_72484_row7_col3" class="data row7 col3" >10,104.00</td>
      <td id="T_72484_row7_col4" class="data row7 col4" >21,186.00</td>
      <td id="T_72484_row7_col5" class="data row7 col5" >58,683.00</td>
      <td id="T_72484_row7_col6" class="data row7 col6" >82,307.00</td>
      <td id="T_72484_row7_col7" class="data row7 col7" >93,132.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row8" class="row_heading level0 row8" >fyear</th>
      <td id="T_72484_row8_col0" class="data row8 col0" >357.00</td>
      <td id="T_72484_row8_col1" class="data row8 col1" >2,018.86</td>
      <td id="T_72484_row8_col2" class="data row8 col2" >0.35</td>
      <td id="T_72484_row8_col3" class="data row8 col3" >2,018.00</td>
      <td id="T_72484_row8_col4" class="data row8 col4" >2,019.00</td>
      <td id="T_72484_row8_col5" class="data row8 col5" >2,019.00</td>
      <td id="T_72484_row8_col6" class="data row8 col6" >2,019.00</td>
      <td id="T_72484_row8_col7" class="data row8 col7" >2,019.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row9" class="row_heading level0 row9" >sic</th>
      <td id="T_72484_row9_col0" class="data row9 col0" >357.00</td>
      <td id="T_72484_row9_col1" class="data row9 col1" >4,218.51</td>
      <td id="T_72484_row9_col2" class="data row9 col2" >1,935.02</td>
      <td id="T_72484_row9_col3" class="data row9 col3" >100.00</td>
      <td id="T_72484_row9_col4" class="data row9 col4" >2,836.00</td>
      <td id="T_72484_row9_col5" class="data row9 col5" >3,728.00</td>
      <td id="T_72484_row9_col6" class="data row9 col6" >5,331.00</td>
      <td id="T_72484_row9_col7" class="data row9 col7" >8,742.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row10" class="row_heading level0 row10" >sic3</th>
      <td id="T_72484_row10_col0" class="data row10 col0" >357.00</td>
      <td id="T_72484_row10_col1" class="data row10 col1" >421.64</td>
      <td id="T_72484_row10_col2" class="data row10 col2" >193.52</td>
      <td id="T_72484_row10_col3" class="data row10 col3" >10.00</td>
      <td id="T_72484_row10_col4" class="data row10 col4" >283.00</td>
      <td id="T_72484_row10_col5" class="data row10 col5" >372.00</td>
      <td id="T_72484_row10_col6" class="data row10 col6" >533.00</td>
      <td id="T_72484_row10_col7" class="data row10 col7" >874.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row11" class="row_heading level0 row11" >td</th>
      <td id="T_72484_row11_col0" class="data row11 col0" >357.00</td>
      <td id="T_72484_row11_col1" class="data row11 col1" >12,090.32</td>
      <td id="T_72484_row11_col2" class="data row11 col2" >21,481.08</td>
      <td id="T_72484_row11_col3" class="data row11 col3" >0.00</td>
      <td id="T_72484_row11_col4" class="data row11 col4" >2,256.90</td>
      <td id="T_72484_row11_col5" class="data row11 col5" >5,135.39</td>
      <td id="T_72484_row11_col6" class="data row11 col6" >12,139.00</td>
      <td id="T_72484_row11_col7" class="data row11 col7" >188,402.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row12" class="row_heading level0 row12" >long_debt_dum</th>
      <td id="T_72484_row12_col0" class="data row12 col0" >357.00</td>
      <td id="T_72484_row12_col1" class="data row12 col1" >0.98</td>
      <td id="T_72484_row12_col2" class="data row12 col2" >0.14</td>
      <td id="T_72484_row12_col3" class="data row12 col3" >0.00</td>
      <td id="T_72484_row12_col4" class="data row12 col4" >1.00</td>
      <td id="T_72484_row12_col5" class="data row12 col5" >1.00</td>
      <td id="T_72484_row12_col6" class="data row12 col6" >1.00</td>
      <td id="T_72484_row12_col7" class="data row12 col7" >1.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row13" class="row_heading level0 row13" >me</th>
      <td id="T_72484_row13_col0" class="data row13 col0" >357.00</td>
      <td id="T_72484_row13_col1" class="data row13 col1" >56,586.11</td>
      <td id="T_72484_row13_col2" class="data row13 col2" >115,922.82</td>
      <td id="T_72484_row13_col3" class="data row13 col3" >4,345.11</td>
      <td id="T_72484_row13_col4" class="data row13 col4" >13,087.14</td>
      <td id="T_72484_row13_col5" class="data row13 col5" >22,488.65</td>
      <td id="T_72484_row13_col6" class="data row13 col6" >51,240.00</td>
      <td id="T_72484_row13_col7" class="data row13 col7" >1,023,856.25</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row14" class="row_heading level0 row14" >l_a</th>
      <td id="T_72484_row14_col0" class="data row14 col0" >357.00</td>
      <td id="T_72484_row14_col1" class="data row14 col1" >9.78</td>
      <td id="T_72484_row14_col2" class="data row14 col2" >1.14</td>
      <td id="T_72484_row14_col3" class="data row14 col3" >6.96</td>
      <td id="T_72484_row14_col4" class="data row14 col4" >8.90</td>
      <td id="T_72484_row14_col5" class="data row14 col5" >9.71</td>
      <td id="T_72484_row14_col6" class="data row14 col6" >10.55</td>
      <td id="T_72484_row14_col7" class="data row14 col7" >13.22</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row15" class="row_heading level0 row15" >l_sale</th>
      <td id="T_72484_row15_col0" class="data row15 col0" >357.00</td>
      <td id="T_72484_row15_col1" class="data row15 col1" >9.39</td>
      <td id="T_72484_row15_col2" class="data row15 col2" >1.15</td>
      <td id="T_72484_row15_col3" class="data row15 col3" >6.60</td>
      <td id="T_72484_row15_col4" class="data row15 col4" >8.55</td>
      <td id="T_72484_row15_col5" class="data row15 col5" >9.27</td>
      <td id="T_72484_row15_col6" class="data row15 col6" >10.02</td>
      <td id="T_72484_row15_col7" class="data row15 col7" >13.15</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row16" class="row_heading level0 row16" >div_d</th>
      <td id="T_72484_row16_col0" class="data row16 col0" >357.00</td>
      <td id="T_72484_row16_col1" class="data row16 col1" >0.79</td>
      <td id="T_72484_row16_col2" class="data row16 col2" >0.41</td>
      <td id="T_72484_row16_col3" class="data row16 col3" >0.00</td>
      <td id="T_72484_row16_col4" class="data row16 col4" >1.00</td>
      <td id="T_72484_row16_col5" class="data row16 col5" >1.00</td>
      <td id="T_72484_row16_col6" class="data row16 col6" >1.00</td>
      <td id="T_72484_row16_col7" class="data row16 col7" >1.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row17" class="row_heading level0 row17" >age</th>
      <td id="T_72484_row17_col0" class="data row17 col0" >357.00</td>
      <td id="T_72484_row17_col1" class="data row17 col1" >0.00</td>
      <td id="T_72484_row17_col2" class="data row17 col2" >0.00</td>
      <td id="T_72484_row17_col3" class="data row17 col3" >0.00</td>
      <td id="T_72484_row17_col4" class="data row17 col4" >0.00</td>
      <td id="T_72484_row17_col5" class="data row17 col5" >0.00</td>
      <td id="T_72484_row17_col6" class="data row17 col6" >0.00</td>
      <td id="T_72484_row17_col7" class="data row17 col7" >0.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row18" class="row_heading level0 row18" >atr</th>
      <td id="T_72484_row18_col0" class="data row18 col0" >357.00</td>
      <td id="T_72484_row18_col1" class="data row18 col1" >0.25</td>
      <td id="T_72484_row18_col2" class="data row18 col2" >0.25</td>
      <td id="T_72484_row18_col3" class="data row18 col3" >0.00</td>
      <td id="T_72484_row18_col4" class="data row18 col4" >0.13</td>
      <td id="T_72484_row18_col5" class="data row18 col5" >0.21</td>
      <td id="T_72484_row18_col6" class="data row18 col6" >0.24</td>
      <td id="T_72484_row18_col7" class="data row18 col7" >1.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row19" class="row_heading level0 row19" >smalltaxlosscarry</th>
      <td id="T_72484_row19_col0" class="data row19 col0" >275.00</td>
      <td id="T_72484_row19_col1" class="data row19 col1" >0.72</td>
      <td id="T_72484_row19_col2" class="data row19 col2" >0.45</td>
      <td id="T_72484_row19_col3" class="data row19 col3" >0.00</td>
      <td id="T_72484_row19_col4" class="data row19 col4" >0.00</td>
      <td id="T_72484_row19_col5" class="data row19 col5" >1.00</td>
      <td id="T_72484_row19_col6" class="data row19 col6" >1.00</td>
      <td id="T_72484_row19_col7" class="data row19 col7" >1.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row20" class="row_heading level0 row20" >largetaxlosscarry</th>
      <td id="T_72484_row20_col0" class="data row20 col0" >275.00</td>
      <td id="T_72484_row20_col1" class="data row20 col1" >0.20</td>
      <td id="T_72484_row20_col2" class="data row20 col2" >0.40</td>
      <td id="T_72484_row20_col3" class="data row20 col3" >0.00</td>
      <td id="T_72484_row20_col4" class="data row20 col4" >0.00</td>
      <td id="T_72484_row20_col5" class="data row20 col5" >0.00</td>
      <td id="T_72484_row20_col6" class="data row20 col6" >0.00</td>
      <td id="T_72484_row20_col7" class="data row20 col7" >1.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row21" class="row_heading level0 row21" >l_emp</th>
      <td id="T_72484_row21_col0" class="data row21 col0" >357.00</td>
      <td id="T_72484_row21_col1" class="data row21 col1" >3.36</td>
      <td id="T_72484_row21_col2" class="data row21 col2" >1.15</td>
      <td id="T_72484_row21_col3" class="data row21 col3" >0.44</td>
      <td id="T_72484_row21_col4" class="data row21 col4" >2.48</td>
      <td id="T_72484_row21_col5" class="data row21 col5" >3.29</td>
      <td id="T_72484_row21_col6" class="data row21 col6" >4.26</td>
      <td id="T_72484_row21_col7" class="data row21 col7" >6.03</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row22" class="row_heading level0 row22" >l_ppent</th>
      <td id="T_72484_row22_col0" class="data row22 col0" >357.00</td>
      <td id="T_72484_row22_col1" class="data row22 col1" >7.99</td>
      <td id="T_72484_row22_col2" class="data row22 col2" >1.46</td>
      <td id="T_72484_row22_col3" class="data row22 col3" >4.85</td>
      <td id="T_72484_row22_col4" class="data row22 col4" >6.90</td>
      <td id="T_72484_row22_col5" class="data row22 col5" >7.91</td>
      <td id="T_72484_row22_col6" class="data row22 col6" >8.97</td>
      <td id="T_72484_row22_col7" class="data row22 col7" >11.11</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row23" class="row_heading level0 row23" >l_laborratio</th>
      <td id="T_72484_row23_col0" class="data row23 col0" >357.00</td>
      <td id="T_72484_row23_col1" class="data row23 col1" >4.70</td>
      <td id="T_72484_row23_col2" class="data row23 col2" >1.41</td>
      <td id="T_72484_row23_col3" class="data row23 col3" >0.51</td>
      <td id="T_72484_row23_col4" class="data row23 col4" >3.83</td>
      <td id="T_72484_row23_col5" class="data row23 col5" >4.39</td>
      <td id="T_72484_row23_col6" class="data row23 col6" >5.39</td>
      <td id="T_72484_row23_col7" class="data row23 col7" >9.93</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row24" class="row_heading level0 row24" >Inv</th>
      <td id="T_72484_row24_col0" class="data row24 col0" >357.00</td>
      <td id="T_72484_row24_col1" class="data row24 col1" >0.05</td>
      <td id="T_72484_row24_col2" class="data row24 col2" >0.09</td>
      <td id="T_72484_row24_col3" class="data row24 col3" >-0.33</td>
      <td id="T_72484_row24_col4" class="data row24 col4" >0.02</td>
      <td id="T_72484_row24_col5" class="data row24 col5" >0.05</td>
      <td id="T_72484_row24_col6" class="data row24 col6" >0.09</td>
      <td id="T_72484_row24_col7" class="data row24 col7" >0.37</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row25" class="row_heading level0 row25" >Ch_Cash</th>
      <td id="T_72484_row25_col0" class="data row25 col0" >357.00</td>
      <td id="T_72484_row25_col1" class="data row25 col1" >0.01</td>
      <td id="T_72484_row25_col2" class="data row25 col2" >0.06</td>
      <td id="T_72484_row25_col3" class="data row25 col3" >-0.32</td>
      <td id="T_72484_row25_col4" class="data row25 col4" >-0.01</td>
      <td id="T_72484_row25_col5" class="data row25 col5" >0.00</td>
      <td id="T_72484_row25_col6" class="data row25 col6" >0.02</td>
      <td id="T_72484_row25_col7" class="data row25 col7" >0.38</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row26" class="row_heading level0 row26" >Div</th>
      <td id="T_72484_row26_col0" class="data row26 col0" >357.00</td>
      <td id="T_72484_row26_col1" class="data row26 col1" >0.03</td>
      <td id="T_72484_row26_col2" class="data row26 col2" >0.03</td>
      <td id="T_72484_row26_col3" class="data row26 col3" >0.00</td>
      <td id="T_72484_row26_col4" class="data row26 col4" >0.00</td>
      <td id="T_72484_row26_col5" class="data row26 col5" >0.02</td>
      <td id="T_72484_row26_col6" class="data row26 col6" >0.04</td>
      <td id="T_72484_row26_col7" class="data row26 col7" >0.14</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row27" class="row_heading level0 row27" >Ch_Debt</th>
      <td id="T_72484_row27_col0" class="data row27 col0" >357.00</td>
      <td id="T_72484_row27_col1" class="data row27 col1" >0.01</td>
      <td id="T_72484_row27_col2" class="data row27 col2" >0.07</td>
      <td id="T_72484_row27_col3" class="data row27 col3" >-0.27</td>
      <td id="T_72484_row27_col4" class="data row27 col4" >-0.02</td>
      <td id="T_72484_row27_col5" class="data row27 col5" >-0.00</td>
      <td id="T_72484_row27_col6" class="data row27 col6" >0.03</td>
      <td id="T_72484_row27_col7" class="data row27 col7" >0.35</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row28" class="row_heading level0 row28" >Ch_Eqty</th>
      <td id="T_72484_row28_col0" class="data row28 col0" >357.00</td>
      <td id="T_72484_row28_col1" class="data row28 col1" >-0.04</td>
      <td id="T_72484_row28_col2" class="data row28 col2" >0.06</td>
      <td id="T_72484_row28_col3" class="data row28 col3" >-0.28</td>
      <td id="T_72484_row28_col4" class="data row28 col4" >-0.06</td>
      <td id="T_72484_row28_col5" class="data row28 col5" >-0.03</td>
      <td id="T_72484_row28_col6" class="data row28 col6" >-0.00</td>
      <td id="T_72484_row28_col7" class="data row28 col7" >0.11</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row29" class="row_heading level0 row29" >Ch_WC</th>
      <td id="T_72484_row29_col0" class="data row29 col0" >357.00</td>
      <td id="T_72484_row29_col1" class="data row29 col1" >0.01</td>
      <td id="T_72484_row29_col2" class="data row29 col2" >0.05</td>
      <td id="T_72484_row29_col3" class="data row29 col3" >-0.25</td>
      <td id="T_72484_row29_col4" class="data row29 col4" >-0.01</td>
      <td id="T_72484_row29_col5" class="data row29 col5" >0.01</td>
      <td id="T_72484_row29_col6" class="data row29 col6" >0.02</td>
      <td id="T_72484_row29_col7" class="data row29 col7" >0.37</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row30" class="row_heading level0 row30" >CF</th>
      <td id="T_72484_row30_col0" class="data row30 col0" >357.00</td>
      <td id="T_72484_row30_col1" class="data row30 col1" >0.12</td>
      <td id="T_72484_row30_col2" class="data row30 col2" >0.07</td>
      <td id="T_72484_row30_col3" class="data row30 col3" >-0.27</td>
      <td id="T_72484_row30_col4" class="data row30 col4" >0.08</td>
      <td id="T_72484_row30_col5" class="data row30 col5" >0.11</td>
      <td id="T_72484_row30_col6" class="data row30 col6" >0.16</td>
      <td id="T_72484_row30_col7" class="data row30 col7" >0.33</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row31" class="row_heading level0 row31" >td_a</th>
      <td id="T_72484_row31_col0" class="data row31 col0" >357.00</td>
      <td id="T_72484_row31_col1" class="data row31 col1" >0.33</td>
      <td id="T_72484_row31_col2" class="data row31 col2" >0.18</td>
      <td id="T_72484_row31_col3" class="data row31 col3" >0.00</td>
      <td id="T_72484_row31_col4" class="data row31 col4" >0.21</td>
      <td id="T_72484_row31_col5" class="data row31 col5" >0.32</td>
      <td id="T_72484_row31_col6" class="data row31 col6" >0.43</td>
      <td id="T_72484_row31_col7" class="data row31 col7" >1.25</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row32" class="row_heading level0 row32" >td_mv</th>
      <td id="T_72484_row32_col0" class="data row32 col0" >357.00</td>
      <td id="T_72484_row32_col1" class="data row32 col1" >0.20</td>
      <td id="T_72484_row32_col2" class="data row32 col2" >0.14</td>
      <td id="T_72484_row32_col3" class="data row32 col3" >0.00</td>
      <td id="T_72484_row32_col4" class="data row32 col4" >0.10</td>
      <td id="T_72484_row32_col5" class="data row32 col5" >0.17</td>
      <td id="T_72484_row32_col6" class="data row32 col6" >0.27</td>
      <td id="T_72484_row32_col7" class="data row32 col7" >0.81</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row33" class="row_heading level0 row33" >mb</th>
      <td id="T_72484_row33_col0" class="data row33 col0" >357.00</td>
      <td id="T_72484_row33_col1" class="data row33 col1" >2.84</td>
      <td id="T_72484_row33_col2" class="data row33 col2" >1.98</td>
      <td id="T_72484_row33_col3" class="data row33 col3" >0.88</td>
      <td id="T_72484_row33_col4" class="data row33 col4" >1.48</td>
      <td id="T_72484_row33_col5" class="data row33 col5" >2.23</td>
      <td id="T_72484_row33_col6" class="data row33 col6" >3.42</td>
      <td id="T_72484_row33_col7" class="data row33 col7" >13.08</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row34" class="row_heading level0 row34" >prof_a</th>
      <td id="T_72484_row34_col0" class="data row34 col0" >357.00</td>
      <td id="T_72484_row34_col1" class="data row34 col1" >0.15</td>
      <td id="T_72484_row34_col2" class="data row34 col2" >0.07</td>
      <td id="T_72484_row34_col3" class="data row34 col3" >0.01</td>
      <td id="T_72484_row34_col4" class="data row34 col4" >0.10</td>
      <td id="T_72484_row34_col5" class="data row34 col5" >0.14</td>
      <td id="T_72484_row34_col6" class="data row34 col6" >0.19</td>
      <td id="T_72484_row34_col7" class="data row34 col7" >0.39</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row35" class="row_heading level0 row35" >ppe_a</th>
      <td id="T_72484_row35_col0" class="data row35 col0" >357.00</td>
      <td id="T_72484_row35_col1" class="data row35 col1" >0.26</td>
      <td id="T_72484_row35_col2" class="data row35 col2" >0.23</td>
      <td id="T_72484_row35_col3" class="data row35 col3" >0.01</td>
      <td id="T_72484_row35_col4" class="data row35 col4" >0.09</td>
      <td id="T_72484_row35_col5" class="data row35 col5" >0.16</td>
      <td id="T_72484_row35_col6" class="data row35 col6" >0.35</td>
      <td id="T_72484_row35_col7" class="data row35 col7" >0.93</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row36" class="row_heading level0 row36" >cash_a</th>
      <td id="T_72484_row36_col0" class="data row36 col0" >357.00</td>
      <td id="T_72484_row36_col1" class="data row36 col1" >0.12</td>
      <td id="T_72484_row36_col2" class="data row36 col2" >0.13</td>
      <td id="T_72484_row36_col3" class="data row36 col3" >0.00</td>
      <td id="T_72484_row36_col4" class="data row36 col4" >0.03</td>
      <td id="T_72484_row36_col5" class="data row36 col5" >0.07</td>
      <td id="T_72484_row36_col6" class="data row36 col6" >0.16</td>
      <td id="T_72484_row36_col7" class="data row36 col7" >0.66</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row37" class="row_heading level0 row37" >xrd_a</th>
      <td id="T_72484_row37_col0" class="data row37 col0" >357.00</td>
      <td id="T_72484_row37_col1" class="data row37 col1" >0.03</td>
      <td id="T_72484_row37_col2" class="data row37 col2" >0.05</td>
      <td id="T_72484_row37_col3" class="data row37 col3" >0.00</td>
      <td id="T_72484_row37_col4" class="data row37 col4" >0.00</td>
      <td id="T_72484_row37_col5" class="data row37 col5" >0.01</td>
      <td id="T_72484_row37_col6" class="data row37 col6" >0.04</td>
      <td id="T_72484_row37_col7" class="data row37 col7" >0.34</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row38" class="row_heading level0 row38" >dltt_a</th>
      <td id="T_72484_row38_col0" class="data row38 col0" >357.00</td>
      <td id="T_72484_row38_col1" class="data row38 col1" >0.29</td>
      <td id="T_72484_row38_col2" class="data row38 col2" >0.17</td>
      <td id="T_72484_row38_col3" class="data row38 col3" >0.00</td>
      <td id="T_72484_row38_col4" class="data row38 col4" >0.18</td>
      <td id="T_72484_row38_col5" class="data row38 col5" >0.28</td>
      <td id="T_72484_row38_col6" class="data row38 col6" >0.39</td>
      <td id="T_72484_row38_col7" class="data row38 col7" >1.07</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row39" class="row_heading level0 row39" >invopps_FG09</th>
      <td id="T_72484_row39_col0" class="data row39 col0" >337.00</td>
      <td id="T_72484_row39_col1" class="data row39 col1" >2.50</td>
      <td id="T_72484_row39_col2" class="data row39 col2" >1.98</td>
      <td id="T_72484_row39_col3" class="data row39 col3" >0.41</td>
      <td id="T_72484_row39_col4" class="data row39 col4" >1.16</td>
      <td id="T_72484_row39_col5" class="data row39 col5" >1.92</td>
      <td id="T_72484_row39_col6" class="data row39 col6" >3.01</td>
      <td id="T_72484_row39_col7" class="data row39 col7" >12.16</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row40" class="row_heading level0 row40" >sales_g</th>
      <td id="T_72484_row40_col0" class="data row40 col0" >0.00</td>
      <td id="T_72484_row40_col1" class="data row40 col1" >nan</td>
      <td id="T_72484_row40_col2" class="data row40 col2" >nan</td>
      <td id="T_72484_row40_col3" class="data row40 col3" >nan</td>
      <td id="T_72484_row40_col4" class="data row40 col4" >nan</td>
      <td id="T_72484_row40_col5" class="data row40 col5" >nan</td>
      <td id="T_72484_row40_col6" class="data row40 col6" >nan</td>
      <td id="T_72484_row40_col7" class="data row40 col7" >nan</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row41" class="row_heading level0 row41" >dv_a</th>
      <td id="T_72484_row41_col0" class="data row41 col0" >357.00</td>
      <td id="T_72484_row41_col1" class="data row41 col1" >0.03</td>
      <td id="T_72484_row41_col2" class="data row41 col2" >0.03</td>
      <td id="T_72484_row41_col3" class="data row41 col3" >0.00</td>
      <td id="T_72484_row41_col4" class="data row41 col4" >0.00</td>
      <td id="T_72484_row41_col5" class="data row41 col5" >0.02</td>
      <td id="T_72484_row41_col6" class="data row41 col6" >0.04</td>
      <td id="T_72484_row41_col7" class="data row41 col7" >0.14</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row42" class="row_heading level0 row42" >short_debt</th>
      <td id="T_72484_row42_col0" class="data row42 col0" >351.00</td>
      <td id="T_72484_row42_col1" class="data row42 col1" >0.11</td>
      <td id="T_72484_row42_col2" class="data row42 col2" >0.12</td>
      <td id="T_72484_row42_col3" class="data row42 col3" >0.00</td>
      <td id="T_72484_row42_col4" class="data row42 col4" >0.03</td>
      <td id="T_72484_row42_col5" class="data row42 col5" >0.08</td>
      <td id="T_72484_row42_col6" class="data row42 col6" >0.15</td>
      <td id="T_72484_row42_col7" class="data row42 col7" >1.00</td>
    </tr>
    <tr>
      <th id="T_72484_level0_row43" class="row_heading level0 row43" >R</th>
      <td id="T_72484_row43_col0" class="data row43 col0" >502.00</td>
      <td id="T_72484_row43_col1" class="data row43 col1" >-0.13</td>
      <td id="T_72484_row43_col2" class="data row43 col2" >0.09</td>
      <td id="T_72484_row43_col3" class="data row43 col3" >-0.61</td>
      <td id="T_72484_row43_col4" class="data row43 col4" >-0.17</td>
      <td id="T_72484_row43_col5" class="data row43 col5" >-0.11</td>
      <td id="T_72484_row43_col6" class="data row43 col6" >-0.07</td>
      <td id="T_72484_row43_col7" class="data row43 col7" >0.12</td>
    </tr>
  </tbody>
</table>




### Summary of the above:

As we can see, we have 56 variables and 505 firms in total. However, we don't have the same amount of observations for all of these variables. For our added risks, we have 489 observations; for returns, we have 502; for most accounting measures, we have 357; however, there are some exceptions like short_debt (351) and smalltaxlosscarry (275). 

What is really important to note for our analysis specifically is that the mean returns are negative (by about 12%), which makes sense as the market crashed during this period of time. Summary analysis of the individual risks has already been covered. Additionally, variables such as gvkey and lpermno aren't relevant for this analysis as they are qualitative measurements but are included so that readers can view the entire dataset.

## Risk - Return Correlations during the week of March 9th - March 13th


```python
corr = firms_df[["Supply_Chain_Risk_1","Supply_Chain_Risk_2","Supply_Chain_Risk_3","Reputation_Risk","Labor_Risk","R"]].corr()

fig, ax = plt.subplots(figsize=(9,9)) # make a big space for the figure
ax = sns.heatmap(corr,
                 center=0,square=True,
                 cmap=sns.diverging_palette(230, 20, as_cmap=True),
                 cbar_kws={"shrink": .5},
                )

#adapted from https://ledatascifi.github.io/ledatascifi-2022/content/03/04e-visualEDA.html?highlight=correlation 
```


    
![png](output_69_0.png)
    


### Key Findings

Every single risk yielded a negative correlation with returns except for Reputation_Risk. This is a little surprising as we would see companies that have higher reputation risk issues as more susceptible to lower returns during an event like COVID-19. However, we can interpret this by saying that big, robust companies tend to be more well-known and thus more concerned with their image, making them talk about this in their 10-k more often. 

This was also, interestingly, the highest correlation out of all displayed. The second "highest" (in absolute terms), is the correlation between Labor_Risk and returns, which indicates that companies that had a lot of international workers tend to struggle the most in terms of returns. This makes sense as we saw that the companies that yielded the most hits in this measurement were in industries that heavily focused on transportation. 

Still, it must be mentioned that no correlation exceeded about 7.5%. These aren't high correlations, indicating that the risk measurements chosen either might not be the most appropriate for this analysis, or weren't analyzed in the most optimal way.

## Risk - Return Correlations during the "collapse period" - Feb 23rd to March 23rd


```python
#collapse week correlation - same as above just for different dates and in one block of code

start_date = datetime(2020, 2, 23) 
end_date = datetime(2020, 3, 23)

stock_returns['date'] = pd.to_datetime(stock_returns['date'].astype(str))

collapse = stock_returns.query('date >= @start_date & date <= @end_date')

collapse['ret'] = pd.to_numeric(collapse['ret'],errors='coerce')

collapse_returns = (collapse
   # compute gross returns for each asset
   .assign(R = 1+collapse['ret'])
   # for each portfolio and time period...
   .groupby(['ticker'])
   # get the gross returns, and cumulate by taking the product
   ['R'].prod()
   # subtract one to get back to simple returns
   -1
)
collapse_returns = collapse_returns.to_frame()

firms_df_collapse = pd.read_csv('output/sp500_accting_plus_textrisks.csv')
firms_df_collapse = firms_df_collapse.merge(collapse_returns, how='left', left_on='Symbol', right_on='ticker', validate='one_to_one')
pd.set_option('display.max_columns', None)
firms_df_collapse

correlation_collapse = firms_df_collapse[["Supply_Chain_Risk_1","Supply_Chain_Risk_2","Supply_Chain_Risk_3","Reputation_Risk","Labor_Risk","R"]].corr()
fig, ax = plt.subplots(figsize=(9,9)) # make a big space for the figure
ax = sns.heatmap(correlation_collapse,
                 center=0,square=True,
                 cmap=sns.diverging_palette(230, 20, as_cmap=True),
                 cbar_kws={"shrink": .5},
                )
correlation_collapse
```

    /var/folders/dt/qsms415d59sdwybzh19f_s7m0000gn/T/ipykernel_57707/430119879.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      collapse['ret'] = pd.to_numeric(collapse['ret'],errors='coerce')





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Supply_Chain_Risk_1</th>
      <th>Supply_Chain_Risk_2</th>
      <th>Supply_Chain_Risk_3</th>
      <th>Reputation_Risk</th>
      <th>Labor_Risk</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Supply_Chain_Risk_1</th>
      <td>1.000000</td>
      <td>0.923299</td>
      <td>0.104409</td>
      <td>0.176055</td>
      <td>0.210854</td>
      <td>0.042555</td>
    </tr>
    <tr>
      <th>Supply_Chain_Risk_2</th>
      <td>0.923299</td>
      <td>1.000000</td>
      <td>0.109375</td>
      <td>0.155716</td>
      <td>0.174183</td>
      <td>0.050587</td>
    </tr>
    <tr>
      <th>Supply_Chain_Risk_3</th>
      <td>0.104409</td>
      <td>0.109375</td>
      <td>1.000000</td>
      <td>0.007779</td>
      <td>0.042480</td>
      <td>0.037299</td>
    </tr>
    <tr>
      <th>Reputation_Risk</th>
      <td>0.176055</td>
      <td>0.155716</td>
      <td>0.007779</td>
      <td>1.000000</td>
      <td>0.198066</td>
      <td>-0.061046</td>
    </tr>
    <tr>
      <th>Labor_Risk</th>
      <td>0.210854</td>
      <td>0.174183</td>
      <td>0.042480</td>
      <td>0.198066</td>
      <td>1.000000</td>
      <td>-0.056396</td>
    </tr>
    <tr>
      <th>R</th>
      <td>0.042555</td>
      <td>0.050587</td>
      <td>0.037299</td>
      <td>-0.061046</td>
      <td>-0.056396</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](output_73_2.png)
    


### Key Findings

We see a completely different picture here. Our supply chain risks are all positively correlated with returns. This is particularly interesting as we now see that over a longer period of time, companies that have supply chains severely exposed to foreign affairs display a positive return over time. We see negative correlations for the other two measurements.

This was somewhat expected as we didn't just look at the week when COVID hit but instead included a time period prior to COVID as well as that deciding week. A lot of companies, such as Apple, crashed at the beginning of the week of March 9th, but then rebounded by March 23rd (the end of our time period here). Perhaps supply chain risk affects firms' returns less on a broader level (outside of a pandemic), and including those times in our analysis could've highlighted this. The other two measurements have a correlation that is negative which is more expected in a risk-return relationship, so looking at a broader time-frame could have brought this to light. Reputation_Risk once again has the "highest" correlation with returns. Finally, the disparity we see in this sample versus the first one might highlight potential issues in the original risk measurement.



## Risk-Return Correlations on Stimulus Day - March 24th



```python
#stimmy day correlation

stimmy_date = datetime(2020, 3, 24) 

stock_returns['date'] = pd.to_datetime(stock_returns['date'].astype(str))

stimmy = stock_returns.query('date == @stimmy_date')

stimmy['ret'] = pd.to_numeric(stimmy['ret'],errors='coerce')

stimmy_returns = (stimmy
   # compute gross returns for each asset
   .assign(R = 1+stimmy['ret'])
   # for each portfolio and time period...
   .groupby(['ticker'])
   # get the gross returns, and cumulate by taking the product
   ['R'].prod()
   # subtract one to get back to simple returns
   -1
)
stimmy_returns = stimmy_returns.to_frame()

firms_df_stimmy = pd.read_csv('output/sp500_accting_plus_textrisks.csv')
firms_df_stimmy = firms_df_stimmy.merge(stimmy_returns, how='left', left_on='Symbol', right_on='ticker', validate='one_to_one')
pd.set_option('display.max_columns', None)
firms_df_stimmy

correlation_stimmy = firms_df_stimmy[["Supply_Chain_Risk_1","Supply_Chain_Risk_2","Supply_Chain_Risk_3","Reputation_Risk","Labor_Risk","R"]].corr()
fig, ax = plt.subplots(figsize=(9,9)) # make a big space for the figure
ax = sns.heatmap(correlation_stimmy,
                 center=0,square=True,
                 cmap=sns.diverging_palette(230, 20, as_cmap=True),
                 cbar_kws={"shrink": .5},
                )
```

    /var/folders/dt/qsms415d59sdwybzh19f_s7m0000gn/T/ipykernel_57707/1423233233.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      stimmy['ret'] = pd.to_numeric(stimmy['ret'],errors='coerce')



    
![png](output_77_1.png)
    


### Key Findings

We see a completely different picture here with all correlations being positive. We expected this as stimulus day would change the way investors behave in the market and thus produce a different return for stocks examined.

## Correlation between Leverage and Return

In efforts to identify a few other measurements that could have affected firms' returns during this time period, I chose to analyze leverage and R&D ratio. With leverage, the idea is that a firm with higher leverage has lower liquidity and might not be able to refinance as easily or take out more debt to deal with challenging market behavior. With R&D, I wanted to see how the level of innovation within firms affected them during the pandemic, with the idea that more innovative companies might be better equipped to adapt to changing trends in the market.

### Week of March 9th - March 13th:


```python
correlation_leverage = firms_df[["td_a","R"]].corr()
correlation_leverage[:1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>td_a</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>td_a</th>
      <td>1.0</td>
      <td>-0.059045</td>
    </tr>
  </tbody>
</table>
</div>



As expected, we have a negative correlation here between total debt/assets and returns, meaning that lower levels of debt in fact did call for higher returns.

### Collapse period - February 23rd - March 23rd:


```python
correlation_leverage_c = firms_df_collapse[["td_a","R"]].corr()
correlation_leverage_c[:1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>td_a</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>td_a</th>
      <td>1.0</td>
      <td>-0.100876</td>
    </tr>
  </tbody>
</table>
</div>



We see the same trend here with the correlation being even stronger than it was above for that given week.

### Stimulus Day - March 24th:


```python
correlation_leverage_s = firms_df_stimmy[["td_a","R"]].corr()
correlation_leverage_s[:1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>td_a</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>td_a</th>
      <td>1.0</td>
      <td>0.062406</td>
    </tr>
  </tbody>
</table>
</div>



Here, we have a slightly different picture - however, stimulus day was bound to affect markets unpredictably so this change in the sign of the correlation isn't too shocking.

## Correlation between Firm R&D Ratio and Return

### Week of March 9th - March 13th:


```python
correlation_rd = firms_df[["xrd_a","R"]].corr()
correlation_rd[:1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>xrd_a</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>xrd_a</th>
      <td>1.0</td>
      <td>0.273802</td>
    </tr>
  </tbody>
</table>
</div>



Here, we actually see that the R&D ratio has a far greater correlation than any of our risk factors or leverage have with returns. It is a positive correlation indicating that firms that invest more in research and development yielded higher returns, proving my hypothesis right (on a very basic level, deeper analysis would need to be done to fully confirm this).

### Collapse period - February 23rd - March 23rd:


```python
correlation_rd_c = firms_df_collapse[["xrd_a","R"]].corr()
correlation_rd_c[:1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>xrd_a</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>xrd_a</th>
      <td>1.0</td>
      <td>0.39534</td>
    </tr>
  </tbody>
</table>
</div>



We see the same trend here with an even higher correlation. This could be due to the fact that prior to when COVID-19 hit, these firms yielded even higher returns, indicating that even in normal times, this factor might very much be a deciding one.

### Stimulus Day - March 24th:


```python
correlation_rd_s = firms_df_stimmy[["xrd_a","R"]].corr()
correlation_rd_s[:1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>xrd_a</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>xrd_a</th>
      <td>1.0</td>
      <td>-0.214656</td>
    </tr>
  </tbody>
</table>
</div>



Again, we see a sign change on stimulus day - an almost expected trend at this point in our analysis.

## Regressions - Analyzing Risk-Return Relationship in a different way


```python
model = sm_ols('R ~ Supply_Chain_Risk_1 + Supply_Chain_Risk_2 + Supply_Chain_Risk_3 + Labor_Risk + Reputation_Risk + td_a + xrd_a', data=firms_df).fit().summary()
model
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>R</td>        <th>  R-squared:         </th> <td>   0.092</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.074</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   4.982</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 25 Mar 2022</td> <th>  Prob (F-statistic):</th> <td>2.18e-05</td>
</tr>
<tr>
  <th>Time:</th>                 <td>01:01:54</td>     <th>  Log-Likelihood:    </th> <td>  326.10</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   351</td>      <th>  AIC:               </th> <td>  -636.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   343</td>      <th>  BIC:               </th> <td>  -605.3</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>           <td>   -0.1473</td> <td>    0.017</td> <td>   -8.730</td> <td> 0.000</td> <td>   -0.180</td> <td>   -0.114</td>
</tr>
<tr>
  <th>Supply_Chain_Risk_1</th> <td>   -0.0013</td> <td>    0.001</td> <td>   -1.436</td> <td> 0.152</td> <td>   -0.003</td> <td>    0.000</td>
</tr>
<tr>
  <th>Supply_Chain_Risk_2</th> <td>    0.0022</td> <td>    0.002</td> <td>    1.142</td> <td> 0.254</td> <td>   -0.002</td> <td>    0.006</td>
</tr>
<tr>
  <th>Supply_Chain_Risk_3</th> <td>   -0.0008</td> <td>    0.001</td> <td>   -0.567</td> <td> 0.571</td> <td>   -0.004</td> <td>    0.002</td>
</tr>
<tr>
  <th>Labor_Risk</th>          <td>-2.506e-05</td> <td>    0.001</td> <td>   -0.022</td> <td> 0.983</td> <td>   -0.002</td> <td>    0.002</td>
</tr>
<tr>
  <th>Reputation_Risk</th>     <td>    0.0057</td> <td>    0.003</td> <td>    2.141</td> <td> 0.033</td> <td>    0.000</td> <td>    0.011</td>
</tr>
<tr>
  <th>td_a</th>                <td>    0.0048</td> <td>    0.029</td> <td>    0.165</td> <td> 0.869</td> <td>   -0.052</td> <td>    0.062</td>
</tr>
<tr>
  <th>xrd_a</th>               <td>    0.6051</td> <td>    0.116</td> <td>    5.203</td> <td> 0.000</td> <td>    0.376</td> <td>    0.834</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>100.453</td> <th>  Durbin-Watson:     </th> <td>   2.050</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 275.056</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.338</td>  <th>  Prob(JB):          </th> <td>1.87e-60</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.412</td>  <th>  Cond. No.          </th> <td>    586.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Regressions here will not be analyzed in great detail, but we can note that the coefficients displayed closely resemble the correlations seen in our prior analysis.

## Additional Interesting Visualizations

### Here is the distribution of returns over the week of March 9th - March 13th.


```python
ax = sns.displot(firms_df, x="R", kde=True, edgecolor="none")
ax.set(title = "S&P 500 returns for the Week of 03/09-03/13")
```




    <seaborn.axisgrid.FacetGrid at 0x7fc644498df0>




    
![png](output_106_1.png)
    


Returns are heavily skewed to the left, meaning that they were mostly negative.

### Here is a visual representation of how the number of hits for all risks correlated to returns.


```python
fig, ax = plt.subplots()
ay = ax.twiny()

ax.scatter(firms_df['Supply_Chain_Risk_1'], firms_df['R'])
ay.scatter(firms_df['Supply_Chain_Risk_2'], firms_df['R'], color='r')
ay.scatter(firms_df['Supply_Chain_Risk_3'], firms_df['R'], color='green')
ay.scatter(firms_df['Labor_Risk'], firms_df['R'], color='yellow')
ay.scatter(firms_df['Reputation_Risk'], firms_df['R'], color='orange')
plt.legend(labels=["Supply_Chain_Risk_2","Supply_Chain_Risk_3","Labor_Risk","Reputation_Risk"])
ax.set(xlabel='Number of hits only for Supply_Chain_Risk_1')
ay.set(xlabel='Number of hits for all other risks')
ax.set(ylabel='Returns')
ax.set(title='Risks versus Return scatterplot')

#adapted from https://stackoverflow.com/questions/59140950/scatter-plot-with-multiple-x-features-and-single-y-in-python
```




    [Text(0.5, 1.0, 'Risks versus Return scatterplot')]




    
![png](output_109_1.png)
    

