# NLP-tools-in-Dash &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.apache.org/licenses/LICENSE-2.0"> <img src="https://www.apache.org/img/asf-estd-1999-logo.jpg" alt="seaborn" style="height:20px; "></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://github.com/plotly"><img src="https://avatars.githubusercontent.com/u/5997976?s=200&v=4" alt="plotly" style="height:20px; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</a> [![PyPI](https://img.shields.io/pypi/v/scikit-learn)](https://pypi.org/project/plotly/)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://pypi.org/project/scikit-learn"><img src="https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo.png" alt="sklearn" style="height:20px; ">[![PyPI](https://img.shields.io/pypi/v/scikit-learn)](https://pypi.org/project/scikit-learn)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://pypi.python.org/pypi/nltk"><img src="https://avatars.githubusercontent.com/u/124114?s=200&v=4" alt="nltk" style="height:20px; "> [![PyPI](https://img.shields.io/pypi/v/scikit-learn)](https://pypi.python.org/pypi/nltk)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://pypi.org/project/seaborn/"><img src="https://avatars.githubusercontent.com/u/22799945?s=200&v=4" alt="seaborn" style="height:20px; "> [![PyPI](https://img.shields.io/pypi/v/scikit-learn)](https://pypi.org/project/seaborn/)

### *A Natural Language Processing (NLP) interactive Plotly Dash tool to process text data - from tokenizing, lemmatizing, etc. all the way to Machine Learning (ML) classification and word prediction.*

### About this app   
#### *NLP analysis in a single app. 11 figures, dropdown and slider analysis controls, ML training and classification*
<img src="https://github.com/semework/NLP-tools-in-Dash/blob/main/assets/images/start.gif" 
style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 75%;" /> 
 
### About dash

* [Dash and how to use it](https://github.com/plotly/dash)
 
Here is a direct quote: 

> Dash is the most downloaded, trusted Python framework for building ML & data science web apps.
Built on top of Plotly.js, React and Flask, Dash ties modern UI elements like dropdowns, sliders, and graphs directly to your analytical Python code. Read our tutorial (proudly crafted ❤️ with Dash itself).

## Getting Started in Python

### Prerequisites and usage

Make sure that dash and its dependent libraries and others listed below are correctly installed (using pip or conda, pip shown here):

```commandline
pip install dash
pip install dash-bootstrap-components
pip install dash-loading-spinners
pip install matplotlib
pip install networkx
pip install nltk
pip install numpy 
pip install pandas
pip install seaborn
pip install wordcloud
pip install yellowbrick
```

Features
--------

* Written entirely in Python - with an interactive ploty Dash web application
* Load text dataframe, parse, tokenize, lemmatize, analyze, train a naive bayes classification model and predict word class.
* Tabbed, interactive and visually-pleasing environment which is easy to use
* Support for doing word relationships using bigram market basket analysis
* Automatic file processing with dropdown for categories and sliders of how many top words (frequency) to plot and display in basket analysis.

Algorithm steps
----------
 
<img src="https://github.com/semework/NLP-tools-in-Dash/blob/main/assets/images/processing_steps.png" 
  style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 85%;" />  

Panels (tabs)
----------
1. DATA & FREQUENCY - has word frequency plots in different formats, and a datatable
2. TREEMAP - Treemap of headline length distributions
3. WORD RELATIONSHIPS - Basket analysis (netowrk and heatmap), top 5 word relationships. Calculated from lemmatized word co-occurence
4. ML (NAIVE BAYES) - detailed freqency distribution for all categories, train and predict words using multinomina naive bayes
----------

1. DATA & FREQUENCY
   
<img src="https://github.com/semework/NLP-tools-in-Dash/blob/main/assets/images/start.gif"   
  style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 75%;"  />

2. TREEMAP
   
<img src="https://github.com/semework/NLP-tools-in-Dash/blob/main/assets/images/treemap.gif"   
  style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 75%;"  />

3. WORD RELATIONSHIPS
   
<img src="[https://github.com/semework/NLP-tools-in-Dash/blob/main/assets/images/wordR.gif" 
  style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 75%;"  />

4. ML (NAIVE BAYES)
   
<img src="https://github.com/semework/NLP-tools-in-Dash/blob/main/assets/images/predict.gif"
  style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 75%;"  />

Controls  

<img src="https://github.com/semework/NLP-tools-in-Dash/blob/main/assets/images/controls.gif"
  style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 75%;"  />

How to use
----------

* Install Python 3.8 or newer and packages mentioned above
* Run the app from the comman line with the python file name followed by the dataframe to use.

  `python3 nlp_dash_tool.py assets/News_Category_Dataset_v3.json`

* use the dropdown and sliders in the first panel (tab) named "DATA & FREQUENCY" to control analysis. 
* The slider for sampling the data is set at 30% by default to give enough data for ML algorithm training

  `self.sample_percent = 30 #percent`

* Use your command-line to follow app loading and analysis results. A few print outs are intentionally added to spy on performance. You will see changes as you play with sliders and the drop down. It will look like this:

```building class...
WELLNESS
Length of all words:  85439
FreqDist:
life     628
time     561
one      557
peopl    539
dtype: int64
...........class built
Dash is running on http://127.0.0.1:9132/

 * Serving Flask app 'nlp_dash_tool'
 * Debug mode: on
```

And if you press "Run model" in "ML (NAIVE BAYES)" tab ~this shows up:

```
Train accuracy score: 84.41%
Test accuracy score: 80.82%
```

After which, if you type in a word to predict, you will see something like this:

```
Your input
tel

Prediction
ENTERTAINMENT

Your input
tele

Prediction
WELLNESS
```

## Documentation

The [Dash](https://dash.plotly.com) contains everything you need to know about the library. It contains useful information of on 
the core Dash components and how to use callbacks, examples, functioning code, and is fully interactive. You can also use the 
[Press & news](https://plotly.com/news/) for a complete and concise specification of the API. 

## More references
* 💻 [Github Repository](https://github.com/plotly/dash) 
* 🗺 [Component Reference](https://dash.plotly.com/reference)

## Contributing and Permissions

Please do not directly copy anything without my concent. Feel free to reach out to me at https://www.linkedin.com/in/mulugeta-semework-abebe/ for ways to collaborate or use some components.

## License

Dash is licensed under MIT. Please view [LICENSE](https://tlo.mit.edu/learn-about-intellectual-property/software-and-open-source-licensing) for more details. For other packages click on corresponding links at the top of this page (first line).

## Acknowledgments

Huge thanks to the following contributors on kaggle. This app would not have been possible without their massive work!

* ❤️ [Thank you AAYUSH JAIN for text classification script](https://www.kaggle.com/code/foolofatook/news-classification-using-bert)
* ❤️ [Thank you YOGESH AGRAWAL for market basket analysis results script](https://www.kaggle.com/code/yugagrawal95/market-basket-analysis-apriori-in-python)
