#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 17:27:00 2023

@author: mulugetasemework
"""
from dash import html, dcc, dash_table
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from math import log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import base64
import dash
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls 
import io
import nltk
import numpy as np
import networkx as nx
import pandas as pd  
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import random
import re
import sys
 
class NLPToolClass:
    def __init__(self, filename):
        print('\n building class...')

        self.filename = filename
        self.figHeight = 160
        self.figWidth = 400
        self.figHeight_wide = 250
        self.figWidth_wide = 700
        self.speed = 1.5
        self.model_run = 0
        self.topW = 10
        self.rules_to_display = 5
        self.grams = 2
        self.tab_height = '50px'
        self.sample_percent = 30 #percent
        self.categories = ['WELLNESS', 'ENTERTAINMENT']

        ############# preprocess data
        self.df_orig = pd.read_json(filename, lines = True)
        self.original_data_size = self.df_orig.shape

        # data for ml, use only specified categories, otherwise it will take too long to train
        # this data is not sampled, as it might not be enough for ml
        self.df_for_ml = self.df_orig.copy()[self.df_orig['category'].isin(self.categories)]
        self.data_prep(self.df_orig, self.sample_percent) 
        self.data_table_title = ("Original data.   " + "   Shape of (row, col):   " + str(self.original_data_size) +
                                " ----------------- Processed and ampled data.   " + "   Shape of (row, col):   " +  str(self.df.shape))

        self.df_counts = pd.DataFrame(self.df['category'].value_counts()).reset_index()
        self.df_counts.columns = ['category','count']
        self.colors = self.generate_colors(len(self.df_counts))


        ###################  process text for first category - upon loading
        ######### subset first category data and set a flag so as not to repeat processing
        
        self.df_tokenized0 = self.process_category(self.categories[0])

        #create baskets for first category
        self.generate_basket(self.df_tokenized0)
        self.update_graphs(self.categories[0], self.topW, self.sample_percent)
 
        # for training and testing naive bayes model, we use all of the original data 
        # i.e. for our specific categories
        # otherwise performance will be bad, or we might not have enough data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_for_ml['short_description'], 
                                            self.df_for_ml['category'], random_state = 13, test_size = 0.3)

        # Get the text categories
        self.text_categories = np.unique(self.df['category'])
 

        ############################
 
        self.panelBackground = 'linear-gradient( 95.2deg, rgba(173,252,234,1) 26.8%, rgba(192,229,246,1) 64%)'

        ##############  build layout ===========
        self.layout = go.Layout(height=500, width=600, margin=dict(t=10,l=10,r=10,b=10))
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                            suppress_callback_exceptions=True)
        self.app.layout = self.build_layout
        print('...........class built')
        
        self.common_styles = {  'display': 'inline-block',
                                'justify-content': 'center', 
                                'align-items': 'center',
                                'margin':'0 auto',
                                'border-top-right-radius': '60px',
                                'border-top-left-radius': '30px',
                                'text-transform': 'lowercase',
                                'border-left': '4px solid gray',
                                'border-right': '2px solid gray',
                                'border-top': '4px solid blue',
                                'fontsize': '9px',
                                'align-items': 'center',
                                'justify-content': 'center',
                                'padding': '0px',
                                'line-height':self.tab_height,
                                'color':'black'}

        self.common_styles_panels = {'overflow': 'scroll', 
                                'margin':'0 auto',
                                'backgroundImage':self.panelBackground,
                                'borderRadius': '12px',
                                'text-align': 'center',
                                'border':'1px black solid', 'justify-content': 'center'}   

        self.fig_pos = {'align-items':'center',
                        'justify-content': 'center',
                        'margin':'0 auto',
                        'display': 'flex',
                        'padding':'10px',
                        'borderRadius': '8px',
                        'overflow': 'scroll'
                        }

        self.centered_image = {  
                        'display': 'inline-block', 
                        'margin':'0 auto',
                        'text-align': 'center',
                        'justify-content': 'center',
                        'align-items':'center', 
                        }

        self.fullPagePanel = {**self.common_styles_panels, 
                                **{'width': '90%', 
                                'height':'60vh',
                                'padding': '40px',
                                'display': 'inline-block', 
                                }}

        self.FullPagePanel_narrow = {**self.common_styles_panels, 
                                **{'width': '90%', 
                                'display': 'flex', 
                                'height':'20vh',
                                }}

        self.QuarterPagePanel ={**self.common_styles_panels, 
                                **{'width': '40%', 
                                'height':'30vh',
                                'overflow': 'scroll', 
                                 }}

        self.OneThirdPagePanel ={**self.common_styles_panels, 
                                **{'width': '30%', 
                                'height':'60vh',
                                }}

        self.HalfPagePanel ={**self.common_styles_panels, 
                                **{'width': '40%', 
                                'height':'55vh',
                                'display': 'inline-block',
                                }}

        self.WideHalfPagePanel ={**self.common_styles_panels, 
                                **{'width': '80%', 
                                'display': 'inline-block', 
                                'height':'55vh',
                                }}

        self.CubicPagePanel ={**self.common_styles_panels, 
                                **{'width': '25%', 
                                'display': 'inline-block', 
                                'height':'60vh',
                                }}

        self.CubicPagePanel_narrow ={**self.common_styles_panels, 
                                **{'width': '80%', 
                                'display': 'inline-block', 
                                'height':'18vh',
                                }}

        self.CubicPagePanel_narrow_text ={'width': '20vh', 
                                'display': 'flex', 
                                'height':'5vh',
                                'margin':'0 auto',
                                'text-align': 'center',
                                'justify-content': 'center','align-items':'center', 
                                }
         
        self.Tabstyle = {**self.common_styles, **{'background-color': '#96c8ff'}} 

        self.selected_style = {'border-top-right-radius': '10px',
                                'border-top-left-radius': '10px',
                                'text-transform': 'uppercase',
                                'border-left': '4px solid gray',
                                'border-right': '2px solid gray',
                                'border-top': '4px solid blue',
                                'fontsize': '10px',
                                'align-items': 'center',
                                'justify-content': 'center',
                                'padding': '0px',
                                'line-height':self.tab_height,
                                'color':'red',
                                'background-color': '#fdfdfd'}
 
        self.tab_style = {'width': '95%', 
                            'display': 'inline-block',
                            'align-items': 'center', 
                            'font-size': '20px',
                            'height':self.tab_height,
                            'padding': '20px',
                            'align-items': 'center',
                            'margin':'50px',
                            'justify-content': 'center'}

        self.app.callback(
                dash.dependencies.Output('class_container', 'children'),
                dash.dependencies.Output('class_container', 'style'),
                dash.dependencies.Input('input1', 'value'),
                dash.dependencies.State('submit_val', 'n_clicks')
            )(self.run_prediction)

        self.app.callback(
                dash.dependencies.Output('train_accuracy_container', 'children'),
                dash.dependencies.Output('test_accuracy_container', 'children'),
                dash.dependencies.Output('confusion_matrix', 'figure'),
                dash.dependencies.Output('confusion_matrix_container', 'style'),
                dash.dependencies.Input('run_model_button', 'n_clicks')
            )(self.run_model)

        self.app.callback(
            [
                dash.dependencies.Output('barG', 'figure'),
                dash.dependencies.Output('pieG', 'figure'),
                dash.dependencies.Output('circleG', 'figure'),
                dash.dependencies.Output('basketG', 'src'),
                dash.dependencies.Output('basket_heatmapG', 'src'),
                dash.dependencies.Output('wordcloudG', 'src'),
                dash.dependencies.Output('treemapG', 'figure'),
                dash.dependencies.Output('data_table_title_container', 'children'),
            ],
            [
            dash.dependencies.Input('category_dropdown', 'value'), 
            dash.dependencies.Input('top_words_slider', 'value'),
            dash.dependencies.Input('sampling_slider', 'value'),
            ],
            )(self.update_graphs)

    def data_prep(self, orig_df, sample_size):
        self.sample_percent = sample_size
        self.df = orig_df.copy().sample(frac=sample_size/100, replace=False)

        # encode (map) categories, if any, into int
        encoder = LabelEncoder()
        self.df['categoryEncoded'] = encoder.fit_transform(self.df['category'])
        # lower case
        self.df['headline'] = self.df.headline.apply(lambda headline: str(headline).lower()).copy()
        self.df['short_description'] = self.df.short_description.apply(lambda descr: str(descr).lower()).copy()
        # headline and description length
        self.df['descr_len'] = self.df.short_description.apply(lambda x: len(str(x).split())).copy()
        self.df['headline_len'] = self.df.copy().headline.apply(lambda x: len(str(x).split())).copy()
        self.df['desc_num'] = self.df['headline_len']  + self.df['descr_len'] 

        self.data_table_title = ("Original data.   " + "   Shape of (row, col):   " + str(self.original_data_size) +
                                " ----------------- Processed and ampled data.   " + "   Shape of (row, col):   " +  str(self.df.shape))

    def process_pos_tag(self, this_df):
        print('in process_pos_tag')
        txt = this_df['lemma_text'][:10].values
        txt = ', '.join(txt)
        self.doc = nlp(txt)
        df = [] 
        for token in self.doc:
            df.append([token.text, token.lemma_, token.pos_, 
                token.tag_, token.dep_,  token.shape_,  
                       token.is_alpha, token.is_stop])
        df = pd.DataFrame(df, columns=['text', 'lemma', 'pos_', 'tag_', 
                        'dep_', 'shape_', 'is_alpha', 'is_stop'])

    def generate_basket(self, df):
        rules0 = []
        for sublist in df.tokenized_text.values:
                bigrams = list(ngrams(sublist, self.grams))
                for grams in bigrams:
                        rules0.append([grams[0],grams[1]])

        rules0 = pd.DataFrame(rules0)
        frq = pd.DataFrame(pd.DataFrame(rules0).value_counts()).reset_index(drop=True)

        rules = pd.concat([rules0, frq], axis=1) 
        rules.columns = ['antecedents', 'consequents','weight']
        rules = rules.copy().sort_values(by=['weight'])

        data = rules.weight.copy()
        rules['norm_weight'] = (data - np.min(data)) / (np.max(data) - np.min(data)) 
        rules['log_weight'] = [log(x) for x in data]
        self.rules = rules.copy()
         
        #pivot basket data
        rules_top=self.rules.copy()[:self.rules_to_display]
        rules_pivot = rules_top.pivot(index='antecedents', columns='consequents', values='norm_weight')
        self.rules_pivot = rules_pivot.round(4)

    def generate_colors(self, color_count):
        Chars = '0123456789ABCDEF'
        random.seed(7)
        colors = ['#'+''.join(random.sample(Chars,6)) for i in range(color_count)]
        return colors

    def hex_to_hls(self, colors_in_hex):
        hlsCols = []
        for c in colors_in_hex:
                RGB = tuple(int(c[i:i+2], 16) for i in (1, 3, 5))
                hlsCols.append("hsl( %d%%,  %d%%, %d%%)" % RGB)
 
        return hlsCols

    def process_text(self, raw_text):
        stemmer = PorterStemmer()  
        wordnet_lem = WordNetLemmatizer()
        letters = re.sub("[^a-zA-Z]", " ", raw_text)   
        words = letters.lower().split()  
        stopWs = set(stopwords.words("english"))   
        good_words = [w for w in words if not w in stopWs]     
        # lemmatized = [wordnet_lem.lemmatize(word) for word in good_words] 
        lemmatized = [stemmer.stem(word) for word in good_words] 
        lemma_df =  ( " ".join(lemmatized))

        return lemma_df
 
    def check_process_category(self, category_name):
        if category_name == self.categories[0]:
            return self.df_tokenized0
        else:
            return self.process_category(category_name)
    
    def process_category(self, category_name):
        df0 = self.df.copy()
        # print(df0.columns, category_name)
        this_df = df0[df0['category']==category_name] 
        del df0     # release memory
        lemmatized_corpus = [self.process_text(text) for text in this_df.short_description] 
        this_df['lemma_text'] = pd.DataFrame(lemmatized_corpus)
        this_df['tokenized_text'] = [word_tokenize(i) for i in lemmatized_corpus]
        this_df['tokenized_freq']=[nltk.FreqDist(i) for i in this_df['tokenized_text']]
        return this_df

    def update_graphs(self, category_name, top_words, sampling_percentage):
        self.sampling_percentage = sampling_percentage
        self.data_prep(self.df_orig, sampling_percentage)
        #this df will be used also for tag processing
        self.df_to_plot = self.check_process_category(category_name)
        self.generate_basket(self.df_to_plot)
        print(category_name)
        all_words  = [x for l in self.df_to_plot['tokenized_text'] for x in l]
        frq = nltk.FreqDist(all_words)
        frq_sorted = {k:v for k,v in sorted(frq.items(), key=lambda item:item[1], reverse=True)}
        freqs =  pd.Series(dict(frq_sorted))

        freqs_df = pd.DataFrame(freqs.items(), columns=['words', 'values'])[:self.topW]
  
        all_words_lem = ' '.join([word for word in all_words])
        print('Length of all words: ', len(all_words))
        print('FreqDist:')
        print(freqs[:4])
 
        x, labels = freqs.index[:self.topW], freqs.index[:self.topW]
        y, values = freqs.values[:self.topW], freqs.values[:self.topW]

        #make tuples for wordcloud
        tuples = [tuple(x) for x in zip(labels,  values)]
        colors = self.generate_colors(len(values))
        hls_colors = self.hex_to_hls(colors)

#################  barG
        barG = {
              'data': [go.Bar(x=values, 
                              y=labels,
                              marker_color= colors,
                              width=[0.4]*len(x),
                              orientation='h',
                              text = labels,
                              )
                       ],
                'layout': go.Layout(
                        autosize=True,
                        font_family='Arial',
                        font_size=16,
                        hovermode='closest',
                        height=self.figHeight_wide,
                        width=self.figWidth_wide+25,
                        yaxis=dict(autorange="reversed"),
                        margin=dict(t=5,l=75,r=5,b=30),
                        )
             }
#################  pieG
        pieG = {
                  'data': [go.Pie(labels=labels, 
                        values=values,
                        marker=dict(colors=colors)
                        )
                           ],
                    'layout': go.Layout(
                        autosize=True,
                        font_family='Arial',
                        font_size=16,
                        hovermode='closest',
                        height=self.figHeight_wide+25,
                        width=self.figWidth_wide,
                        margin=dict(t=5,l=5,r=5,b=5),
                        )
                 }

#################  basket heatmap
        buf = io.BytesIO()  #in-memo file
        plt.figure()
        # create mask
        mask = np.tril(np.ones_like(self.rules_pivot))
         
        # plot triangle heatmap
        sns.heatmap(self.rules_pivot, cmap="YlGnBu", annot=True, mask=mask,
                               fmt="",cbar_kws={'label': 'Normalized count'},
                               linewidths=2, linecolor='white')
        plt.xlabel('Consequent', fontsize=14);
        plt.ylabel('Antecedent', fontsize=14);
        plt.yticks(rotation= 0) 
        plt.xticks(rotation= 45) 
        plt.axis('on')
        plt.savefig(buf, format = "png", bbox_inches='tight', transparent=True)
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("utf8")
        buf.close()
        basket_heatG = "data:image/png;base64,{}".format(data)

#################  wordcloud
        wordcloud = WordCloud(
                        height=self.figHeight_wide-40,
                        width=self.figWidth_wide,
                        random_state=2, 
                        max_font_size=100,
                         ).generate_from_frequencies(dict(tuples))

        buf = io.BytesIO()  #in-memo file
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(buf, format = "png", bbox_inches='tight', transparent=True)
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("utf8")
        buf.close()
        wordcloudG = "data:image/png;base64,{}".format(data)

#################  circleG
        n=30
        pal = list(sns.color_palette(palette='Reds_r', n_colors=n).as_hex())
        fig = px.pie(freqs_df, values= 'values', names='words', color_discrete_sequence=colors)
        fig.update_traces(textposition='outside', textinfo='percent+label', 
                hole=0.2, hoverinfo='percent+label+name')
        circleG = fig.update_layout(width=self.figHeight_wide*2, height=self.figHeight_wide, 
                font=dict(size=16), 
                margin=dict(l=40,r=40, t=5, b=5))
        
#################  basket
        buf = io.BytesIO()  #in-memo file
        plt.figure()
        # fig_basket, ax = plt.subplots()
        G1 = nx.DiGraph()
        color_map=[]
        N = 50
        colors_basket = np.random.rand(N)    
        strs=['Rule 0', 'Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5', 
          'Rule 6', 'Rule R7', 'Rule R8', 'Rule 9', 'Rule 10', 'Rule 11']

        for i in range(self.rules_to_display):

                G1.add_nodes_from(["Rule "+str(i)])
                for a in self.rules.iloc[i]['antecedents']:
                    G1.add_nodes_from([self.rules.iloc[i]['antecedents']])
                    G1.add_edge(self.rules.iloc[i]['antecedents'], "Rule "+str(i), color=colors_basket[i],  
                                weight = 20)

                for c in self.rules.iloc[i]['consequents']:
                    G1.add_nodes_from([self.rules.iloc[i]['consequents']])
                    G1.add_edge("Rule "+str(i), self.rules.iloc[i]['consequents'], color=colors_basket[i],
                                weight = 20)

        for node in G1:
                found_a_string = False
                for item in strs: 
                    if node==item:
                        found_a_string = True
                if found_a_string:
                    color_map.append('yellow')
                else:
                    color_map.append('green')       

        edges = G1.edges()
        colors_basket = [G1[u][v]['color'] for u,v in edges]
        weights = [G1[u][v]['weight'] for u,v in edges]

        d = dict(G1.degree)
        pos = nx.spring_layout(G1, k=30, scale=1)
        nx.draw_networkx(G1, pos, 
                alpha=0.9,
                node_color = color_map, 
                width=[v*2  for v in d.values()],
                font_size=16, 
                edge_color='crimson',
                node_size=[log(x)*5000  for x in d.values()],
                with_labels=True,)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(buf, format = "png", bbox_inches='tight', transparent=True)
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("utf8") #  
        buf.close()
        basketG = "data:image/png;base64,{}".format(data)

#################  treemap
        fig = px.treemap(self.df,
                         path=['category', 'descr_len'],
                         values='headline_len')

        treemapG = fig.update_layout(
                width=self.figWidth_wide*2.5,height=self.figHeight_wide*2,
                font=dict(size=20), template=None,
                margin=dict(t=25,l=25,r=25,b=25))

                # Orders:
                        # barG
                        # pieG
                        # circleG
                        # basketG
                        # basket_heatmapG
                        # wordcloudG
                        # treemapG

        return [barG, pieG, circleG, basketG, basket_heatG, wordcloudG, treemapG, f"{self.data_table_title}"]

    # word or sentence predictor
    def run_prediction(self, sentence, n_clicks):
        if self.model_run != 0:
            all_categories_names = np.array(self.text_categories)
            prediction = self.model.predict([sentence])[0]
            print()
            print("Your input")
            print(sentence)
            print()
            print("Prediction")
            print(prediction)
 
            return [str(prediction), {'background-color':'cyan'}]
        else:
            return ["Please train model first by clicking on 'Run model' button",{'background-color':'yellow'}]

    def run_model(self, n_clicks):
        if n_clicks > 0:
          # Build the model
            self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
            # Train the model using the training data
            self.model.fit(self.X_train, self.y_train)

            nb_pred_train = self.model.predict(self.X_train)
            self.train_score = accuracy_score(self.y_train, nb_pred_train) * 100
            print(f"Train accuracy score: {self.train_score:.2f}%")

            nb_pred_test = self.model.predict(self.X_test)
            self.test_score = accuracy_score(self.y_test, nb_pred_test) * 100
            print(f"Test accuracy score: {self.test_score:.2f}%")
            
            self.model_run = 1

            cm = confusion_matrix(self.y_test, nb_pred_test)
     
            x = self.categories
            y = self.categories

            # change each element of cm to type string for annotations
            cm_text = [[str(y) for y in x] for x in cm]

            # group_names = ['True Neg','False Pos','False Neg','True Pos']
            group_names = ['TP','FP','FN','TP']
            group_counts = ["{0:0.0f}".format(value) for value in
                            cm.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in
                                 cm.flatten()/np.sum(cm)]
            labels = [f"({v1}) ({v2})  {v3}" for v1, v2, v3 in
                      zip(group_names, group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(2,2)
            fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=labels, colorscale='Blues')

            # add title
            fig.update_layout(
                               xaxis = dict(title='Annotations: (quadrant class) (count) percentage'),
                             )
            # add custom xaxis title
            fig.add_annotation(dict(font=dict(color="black",size=14),
                                    x=0.5,
                                    y=-0.15,
                                    showarrow=False,
                                    text="Predicted value",
                                    xref="paper",
                                    yref="paper"))

            # add custom yaxis title
            fig.add_annotation(dict(font=dict(color="black",size=14),
                                    x=-0.35,
                                    y=0.5,
                                    showarrow=False,
                                    text="Real value",
                                    textangle=-90,
                                    xref="paper",
                                    yref="paper"))

            # add annotatoin text key
            fig.add_annotation(dict(font=dict(color="red",size=15),
                                    x=0.5,
                                    y=-0.25,
                                    showarrow=False,
                                    text="Annotations: (quadrant class)   (count)   percentage",
                                    xref="paper",
                                    yref="paper"))

            # adjust margins to make room for yaxis title
            fig.update_layout(margin=dict(t=25,l=25,r=25,b=25))

            # add colorbar
            fig['data'][0]['showscale'] = True
            
            return  [f"{self.train_score:.2f}%", 
                    f"{self.test_score:.2f}%", fig, {'display':'block'}]
        else:
            return '','',{}, {'display':'none'}

    def build_layout(self):
        layout = html.Div(

            children = [
                   html.Div(
                        [
                            html.H1(children='NLP tool in Plotly Dash',
                                style={'width':'100%','display':'inline-black', 'margin':'0 auto',
                                        'background': '#96c8ff','color':'black',
                                        'textAlign': 'center','border':'none', 
                                        }), 

            dcc.Tabs(id="NLP_tabs", value='frequency_graphs', 
                style=self.tab_style,
                children=[

        ### freq tab #########################

                dcc.Tab(label='DATA & FREQUENCY', value='frequency_graphs', 
                    style=self.Tabstyle, selected_style = self.selected_style,
                        children=[
                                    html.Div(                                     
                                           children=[ 
                                            html.Div(
                                                        children=[
                                                                html.Br(),
                                                                html.H6('Category',
                                                                style={'text-align':"right", "color":"red"}),

                                                                html.P(['→'], style={'color': 'black', 'margin-left':'5px',
                                                                'margin-right':'2px', 'fonts-size':'25px'}),

                                                                dcc.Dropdown(self.categories,
                                                                        self.categories[0], #this ideally should be a place holder
                                                                        # but it is needed here to draw the first figures,
                                                                        # otherwise graph space will be empty until
                                                                        # a dropdown selection is made
                                                                        id='category_dropdown',
                                                                        style={ 'width': '275px',
                                                                        'padding': '1px',
                                                                        'font-family':'Arial, Helvetica, sans-serif',
                                                                        'fonts-size':'13px',
                                                                        'font-weight':'200',
                                                                        'border-radius':'5px',
                                                                        '-webkit-appearance':'none',
                                                                        '-moz-appearance':'none',
                                                                        'text-indent':'0.01px',
                                                                        'background':'#f8f9f5',
                                                                        'color':'black'}
                                                                    ),#end dropdown

                                                                   html.H6('Top N words', 
                                                                           style={'text-align':"right", "margin-left" :"10%", 
                                                                                  'color':'red'}),

                                                                    html.P(['→'], style={'color': 'black', 
                                                                     'fonts-size':'25px', 'margin-left':'5px'}),

                                                                    html.Div([
                                                                                dcc.Slider(0, 100, value=self.topW,  step=1,
                                                                                             tooltip = {'placement': 'bottom', 'always_visible': True},
                                                                                             marks={
                                                                                                    25: '25',
                                                                                                    50: '50',
                                                                                                    75: '75',
                                                                                                    100: '100'
                                                                                                    },
                                                                                              id = 'top_words_slider'
                                                                                          ),

                                                                                        ], style={"width": "250px", 'border':'1px black solid',
                                                                                        'color':'red',
                                                                                            'border-radius': '5px', 'padding': '1px'},
                                                                                     ),#end slider html.div

                                                                html.H6('Sampling percentage', 
                                                                           style={'text-align':"right", "margin-left" :"10%", 
                                                                                  'color':'red'}),

                                                                    html.P(['→'], style={'color': 'black', 
                                                                     'fonts-size':'25px', 'margin-left':'5px'}),

                                                                    html.Div([
                                                                                dcc.Slider(0, 100, value=self.sample_percent,  step=1,
                                                                                             tooltip = {'placement': 'bottom', 'always_visible': True},
                                                                                             marks={
                                                                                                    25: '25%',
                                                                                                    50: '50%',
                                                                                                    75: '75%',
                                                                                                    100: '100%'
                                                                                                    },
                                                                                              id = 'sampling_slider'
                                                                                          ),

                                                                                        ], style={"width": "250px", 'border':'1px black solid',
                                                                                        'border-radius': '5px', 'padding': '1px'},
                                                                                     ),#end slider html.div

                                                                        html.Div([

                                                                                html.A(
                                                                                                html.Button('Reset app',
                                                                                                        style={"width": "150px", 
                                                                                                        'box-shadow':'0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19)',
                                                                                                        'border':'2px red solid',
                                                                                                        'border-radius': '15px', 
                                                                                                        'background': 'blue',
                                                                                                        'color': 'white',
                                                                                                        "font-size": "25px"},
                                                                                                        ),  
                                                                                               href='/',
                                                                                                ),

                                                                                  ], style={"width": "250px", 
                                                                                        'border':'none',
                                                                                        'padding': '1px'
                                                                                        },

                                                                                     ),#end slider html.div
                                                        ],

                                                                style={"justify-content": "center", 
                                                                        "display": "flex",
                                                                        "font-size": "16px",
                                                                        "border": "none",
                                                                        'outline':"none",
                                                                        'background-color': "inherit",
                                                                        "font-family": 'inherit',
                                                                        "color":"black",
                                                                        'margin':"0 auto",
                                                                        'padding': "0px",
                                                                        'width':'90%',
                                                                        'text-align': 'center',
                                                                        'justify-content': 'center',
                                                                        },
                                                                ),#end drop div
                                                    ], 
                                                    style={"color":"red"}
                                            ), 

         html.Br(), 
                        dbc.Row(
                            children=[


                                html.Div(
                                        children=[
                                                        html.H3('Word frequencies, top '+ str(self.topW) + ' shown'),
                                                        
                                                ],style =self.centered_image,
                                        ),
                                        html.Br(), 
        ######################## bar

                                            dbc.Col(
                                                html.Div(  
                                            children=[
                                                        html.Div(
                                                                children=[
                                                                    html.H4("Bar"),
                                            dcc.Loading(
                                                        dls.Hash(    
                                                            html.Div(id="barG_container", 
                                                                children= [
                                                                    dcc.Graph(
                                                                        id="barG",
                                                                        figure={
                                                                            "data": [
                                                                                     ],
                                                                            "layout": {
                                                                                "height": '500',
                                                                                "height": '500',  
                                                                                'margin':'dict(l=10,r=10, t=10, b=10)', 
                                                                                     },
                                                                                 },
                                                                          ),
                                                                     ], style =self.centered_image,
                                                                     ),
                                                                color="#435278", 
                                                                speed_multiplier=self.speed,
                                                                size=100),
                                                        ),
                                                                            ],  
                                                                            ),
                                                        ],#children      
                                                    ),
                                                    md=4,          
                                                      style=self.QuarterPagePanel
                                                    ),

        ######################## pie

                                        dbc.Col(
                                                html.Div(  
                                                children=[
                                                html.Div(
                                                    children=[
                                                        html.H4("Pie"),
                                                    dcc.Loading(
                                                        dls.Hash(    
                                                            html.Div(id="pieG_container", 
                                                                children= [
                                                                    dcc.Graph(
                                                                        id="pieG",
                                                                        figure={
                                                                            "data": [
                              
                                                                                     ],
                                                                            "layout": {
                                                                                "height": '550', 
                                                                                'margin':'dict(l=10,r=10, t=10, b=10)', 
                                                                                     },
                                                                                 },
                                                                          ),
                                                                     ], style =self.centered_image,
                                                                     ),
                                                                color="#435278", 
                                                                speed_multiplier=self.speed,
                                                                size=100),
                                                        ),
                                                                            ],  
                                                                            ),
                                                        ],#children      
                                                    ),
                                                md=4,        
                                                style=self.QuarterPagePanel
                                                    ),

        ######################## circle

                                        dbc.Col(
                                                html.Div(  
                                                children=[
                                                    html.Div(
                                                            children=[
                                                        html.H4("Circle"),
                                                    dcc.Loading(
                                                        dls.Hash(    
                                                            html.Div(id="circleG_container", 
                                                                children= [
                                                                    dcc.Graph(
                                                                        id="circleG",
                                                                          ),
                                                                     ], style =self.centered_image,
                                                                     ),
                                                                color="#435278", 
                                                                speed_multiplier=self.speed,
                                                                size=100),
                                                        ),
                                                                            ],  
                                                                            ),
                                                        ],#children      
                                                    ),
                                                md=4,        
                                                style=self.QuarterPagePanel
                                                    ),

        ######################## wordcloud

                                        dbc.Col(
                                                html.Div(  
                                                children=[
                                                    html.Div(
                                                            children=[
                                                                html.H4("Wordcloud"),
                                                    dcc.Loading(
                                                        dls.Hash(    
                                                            html.Div(id="wordcloud_container", 
                                                                children= [
                                                                    html.Img(
                                                                        id="wordcloudG",
                                                                          ),
                                                                     ], style =self.centered_image,
                                                                     ),
                                                                color="#435278", 
                                                                speed_multiplier=self.speed,
                                                                size=100),
                                                        ),
                                                                            ],  
                                                                            ),
                                                        ],#children      
                                                    ),
                                                md=4,         
                                                style=self.QuarterPagePanel
                                                    ),
                            ]
                        ),

                        html.Br(),  
         
                        dbc.Row(
                             [
                            ]
                        ),

                        html.Br(),  
                        dbc.Row(
                                [
                                        html.Div(
                                                children=[
                                                         html.Div(id='data_table_title_container',
                                                                children='',
                                                                style={'margin':'0 auto',
                                                                'width':'90%',
                                                                'height':'10%',
                                                                'color':'black',
                                                                'font-size':'20px'
                                                                }
                                                                ),
                                                                ],style =self.centered_image,
                                                        ),

                                    html.Div(
                                        
                                            children=[
                                                        dash_table.DataTable(id='original_data',
                                                                    data=self.df.to_dict('records'),
                                                                    columns=[{'id': c, 'name': c} for c in self.df.columns],
                                                                    style_cell_conditional=[
                                                                        {
                                                                            'if': {'column_id': c},
                                                                            'textAlign': 'left'
                                                                        } for c in ['Date', 'Region']
                                                                    ],
                                                                    style_data={
                                                                        'color': 'black',
                                                                        'border': '1px solid blue',
                                                                        'backgroundColor': 'white',
                                                                        'whiteSpace': 'normal', 'height': 'auto',
                                                                    },
                                                                    style_data_conditional=[
                                                                        {
                                                                            'if': {'row_index': 'odd'},
                                                                            'backgroundColor': 'rgb(220, 220, 220)',
                                                                        }
                                                                    ],
                                                                    style_header={
                                                                        'backgroundColor': 'rgb(210, 210, 210)',
                                                                        'color': 'black',
                                                                        'fontWeight': 'bold',
                                                                        'border': '1px solid pink',
                                                                        'fontWeight ': 'bold',
                                                                        'font-size': '16px',
                                                                        'text-align': 'center'
                                                                        },
                                                                sort_action='native',
                                                                fill_width = True,
                                                                style_table={
                                                                        'overflowX': 'auto',
                                                                        'overflowY': 'auto', 
                                                                        "border" :"2px black solid",
                                                                        'width':'100%',
                                                                        'height': '68vh',
                                                                        'margin': '0 auto',
                                                                        'padding': '20px',},
                                                                row_deletable=False,
                                                                page_size = 20,  
                                                                )
                                                    ],#style=self.FullPagePanel_narrow  
                                                    ),
                                    ],
                                    ),
                        ], 
                    ),

         ### Treemap #########################

                dcc.Tab(label='TREEMAP',  value='treemap',
                        style = self.Tabstyle, selected_style = self.selected_style,
                        children=[
                            html.Br(),
                        dbc.Row(
                            [
                                html.Div(
                                        children=[
                                                        html.H3(['Treemap of headline length distributions.', html.Br(), 
                                                        'Hover over cells to see a description']),
                                                ],style =self.centered_image,
                                        ),
                                        html.Br(), html.Br(),

                                        dbc.Col(
                                                html.Div(  
                                                children=[
                                                    html.Div(
                                                            children=[
                                                    dcc.Loading(
                                                        dls.Hash(    
                                                            html.Div(id="treemap_container", 
                                                                children= [
                                                                    dcc.Graph(
                                                                        id="treemapG",
                                                                        figure={
                                                                            "data": [  
                                                                                     ],
                                                                                 },
                                                                          ),
                                                                     ], style = self.centered_image,
                                                                     ),
                                                                color="#435278", 
                                                                speed_multiplier=self.speed,
                                                                size=100),
                                                        ),
                                                                            ],  style ={  
                                        'display': 'flex', 
                                        'margin':'0 auto',
                                        'text-align': 'center',
                                        'justify-content': 'center',
                                        'align-items':'center', 
                                        'width':'95%', 
                                        },
                                                                            ),
                                                        ],#children      
                                                    ),
                                                    md=4,          
                                                      style=self.fullPagePanel
                                                    ),
                            ]
                        ),
                        ],
                    ),

         ### Word relationship #########################

                dcc.Tab(label='WORD RELATIONSHIPS',  value='basket_heatmap',
                        style = self.Tabstyle, selected_style = self.selected_style,
                        children=[
                html.Br(),  
                        dbc.Row(
                            [
        ######################## basket_heatmapG
                                        
                                                html.Div(
                                                        children=[
                                                                html.H3('Basket analysis, top '+ str(self.rules_to_display) + 
                                                                        ' word relationships. Calculated from lemmatized word co-occurence'),
                                                                ],style =self.centered_image,
                                                        ),
                                                html.Br(),  html.Br(),
                                        dbc.Col(
                                                html.Div(  
                                                children=[
                                                            html.Div(
                                                                    children=[
                                                                     html.H3('Network'),
                                                                     html.H4('Line width depicts relationship strength'),
                                                    dcc.Loading(
                                                        dls.Hash(    
                                                            html.Div(id="basketG_container", 
                                                                children= [
                                                                    html.Img(
                                                                        id="basketG",
                                                                          ),
                                                                     ], style =self.centered_image,
                                                                     ),
                                                                color="#435278", 
                                                                speed_multiplier=self.speed,
                                                                size=100),
                                                        ),
                                                                            ],  
                                                                            ),

                                                        ],#children      
                                                    ),
                                                    md=4,           
                                                      style=self.HalfPagePanel
                                                    ),

        ######################## basket_heatmapG

                                        dbc.Col(
                                                html.Div(  
                                                children=[
                                                    html.Div(
                                                            children=[
                                                                html.H3('Heatmap'),
                                                    dcc.Loading(
                                                        dls.Hash(    
                                                            html.Div(id="basket_heatmapG_container", 
                                                                children= [
                                                                    html.Img(
                                                                        id="basket_heatmapG",
                                                                          ),
                                                                     ], style =self.centered_image,
                                                                     ),
                                                                color="#435278", 
                                                                speed_multiplier=self.speed,
                                                                size=100),
                                                        ),
                                                                            ],  
                                                                            ),
                                                        ],#children      
                                                    ),
                                                md=4,          
                                                style=self.HalfPagePanel
                                                    ), 
                            ]
                        ),
                        ],

                    ),

         ### classification #########################

                dcc.Tab(label='ML (NAIVE BAYES)',  value='naive_bayes',
                        style = self.Tabstyle, selected_style = self.selected_style,
    
                        children=[
                html.Br(),  
                        dbc.Row(
                            [
        ######################## description
                                        
                                        dbc.Col(
                                            html.Div(  
                                            children=[
                                             html.Div(
                                              children=[
                                #####################################   Short description                                      
                                                html.H4("Short description & headline number of Words"),
                                                    dcc.Loading(
                                                        dls.Hash(    
                                                              html.Div(
                                                                    children=[
                                                                      dcc.Graph(
                                                                       id="description_bar",
                                                                        figure={
                                                                          'data': [
                                                                            go.Histogram(x=self.df['desc_num'],
                                                                                ) 
                                                                                   ],
                                                                            'layout': go.Layout(
                                                                                xaxis_title="Samples with unique description + headline",
                                                                                yaxis_title="Counts",
                                                                                autosize=True,
                                                                                font_family='Arial',
                                                                                font_size=14,
                                                                                hovermode='closest',
                                                                                height=self.figHeight,
                                                                                width=self.figWidth+50,
                                                                                margin=dict(l=50,r=30, t=10, b=30),
                                                                                    )
                                                                         }
                                                                      ),   
                                                                     ], style=self.fig_pos,
                                                                     ),
                                                                color="#435278", 
                                                                speed_multiplier=self.speed,
                                                                size=100),
                                                        ),

                            #####################################   Category distributions  

                                        html.H4("Category distributions"),
                                                    dcc.Loading(
                                                        dls.Hash(    
                                                                html.Div(
                                                                children=[                                     
                                                                   html.H5("Bar"), 
                                                                    dcc.Graph(
                                                                        id="dist_bar",
                                                                        figure={
                                                                              'data': [go.Bar(x=self.df_counts['category'], 
                                                                                              y=self.df_counts['count'],
                                                                                              marker_color=self.colors[:len(self.df_counts)],
                                                                                              width=[0.4]*len(self.df_counts),
                                                                                              )
                                                                                       ],
                                                                                'layout': go.Layout(
                                                                                        autosize=True,
                                                                                        font_family='Arial',
                                                                                        font_size=12,
                                                                                        hovermode='closest',
                                                                                        height=self.figHeight,
                                                                                        width=self.figWidth+50,
                                                                                        margin=dict(l=50,r=10, t=10, b=60),
                                                                                        )
                                                                             }
                                                                          ),  
                                                                             ],style=self.fig_pos,
                                                                             ),
                                                                color="#435278", 
                                                                speed_multiplier=self.speed,
                                                                size=100),
                                                        ),

                            #####################################   Pie
                                                    
                                                    dcc.Loading(
                                                        dls.Hash(    
                                                              html.Div(
                                                                    children=[
                                                                    html.H5("Pie"),
                                                                        dcc.Graph(id='dist_pie0',
                                                                          figure={
                                                                              'data': [go.Pie(labels=self.df_counts['category'], 
                                                                                              values=self.df_counts['count'],
                                                                                             marker=dict(colors=self.colors[:len(self.df_counts)])
                                                                                              )
                                                                                       ],
                                                                                'layout': go.Layout(
                                                                                        autosize=True,
                                                                                        font_family='Arial',
                                                                                        font_size=12,
                                                                                        hovermode='closest',
                                                                                        height=self.figHeight,
                                                                                        width=self.figWidth+50,
                                                                                        margin=dict(l=25,r=30, t=10, b=50),
                                                                                        )
                                                                             }
                                                                          ),  
                                                                     ], style=self.fig_pos,
                                                                     ),
                                                                color="#435278", 
                                                                speed_multiplier=self.speed,
                                                                size=100),
                                                        ),
                                                                            ],  
                                                                            ),
                                                        ],#children      
                                                    ),
                                                    md=4,           
                                                      style=self.OneThirdPagePanel
                                                    ),

        ######################## run model 

                                        dbc.Col(
                                                html.Div(  
                                                children=[
                                                        html.Div( 
                                                            html.Button('Run model', id='run_model_button', n_clicks=0,
                                                                                    style={
                                                                                            'justify-content':'center',
                                                                                            'borderRadius': '5px',
                                                                                            'border':'2px gray solid',
                                                                                            'color':'white',
                                                                                            'background-color':'blue',
                                                                                            'margin':'5px',
                                                                                            'font-size':'20px'
                                                                                            },
                                                                                                ), 
                                                                        style=self.CubicPagePanel_narrow_text,
                                                                                            ),

                                                            html.Div(
                                                            children=[
                                                             html.H4("Confusion matrix"),
                                                                dls.Hash(    
                                                                    html.Div(id="confusion_matrix_container", 
                                                                        children= [
                                                                    dcc.Graph(
                                                                        id="confusion_matrix",
                                                                        figure={
                                                                            "data": [
                              
                                                                                     ],
                                                                            "layout": {
                                                                                "height": '550', 
                                                                                'margin':'dict(l=10,r=10, t=10, b=10)', 
                                                                                     },
                                                                                 },
                                                                          ),
                                                                     ]
                                                                     ),
                                                                color="#435278", 
                                                                speed_multiplier=self.speed,
                                                                size=100),
                                                                                        ],  
                                                                                        ),
                                                        ],#children      
                                                    ),
                                                md=4,          
                                                style=self.OneThirdPagePanel
                                                    ), 

        ######################## accuracies

                                        dbc.Col(
                                                html.Div(  
                                                children=[
                                                 html.Div(
                                                  children=[
                                                    html.H4("Accuracies"),
                                                       
                                                html.H5(f"Train accuracy score:"),
                                                    dls.Hash(    
                                                         html.Div(id='train_accuracy_container',
                                                                children=' ',
                                                            style={'margin':'0 auto',
                                                                'width':'20%',
                                                                'height':'10%',
                                                                'color':'red',
                                                                'font-size':'20px'
                                                                }
                                                                ),
                                                    color="#435278", 
                                                    speed_multiplier=self.speed,
                                                    size=30),

                                                html.H5(f"Test accuracy score:"),

                                                dls.Hash( 
                                                           html.Div(id='test_accuracy_container',
                                                                children=' ',
                                                            style={'margin':'0 auto',
                                                                'width':'20%',
                                                                'height':'10%',
                                                                'color':'red',
                                                                'font-size':'20px'
                                                                }
                                                                ),

                                                        color="#435278", 
                                                    speed_multiplier=self.speed,
                                                    size=30),

                                                        html.Br(), html.Br(),

                                                        html.Div( 
                                                           children=[
                                                              html.H4("Fun with words"),
                                                              html.H5("Type in a word you'd like to classify"),
                                                                dcc.Input(id="input1", type="text", 
                                                                    placeholder="type or paste here...", 
                                                                    style={'margin':'0 auto',
                                                                            'width':'60%',
                                                                            'height':'20%',
                                                                            }
                                                                            ),

                                                                html.Button('Classify', id='submit_val', n_clicks=0,
                                                                        style={
                                                                                'justify-content':'center',
                                                                                'borderRadius': '5px',
                                                                                'border':'2px gray solid',
                                                                                'color':'white',
                                                                                'background-color':'blue',
                                                                                'margin':'5px',
                                                                                'font-size':'20px'
                                                                                }
                                                                                    ),
                                                                            ], style=self.CubicPagePanel_narrow,
                                                                            ),


                                                            html.Br(), html.Br(),  
                                                                html.Div( 
                                                                    children=[
                                                                      html.H4("Classification results"),
                                                                 
                                                                       html.Div(id='class_container',
                                                                            children=' ',
                                                                        style={'margin-top':'5px',
                                                                            'margin':'0 auto',
                                                                            'width':'90%',
                                                                            'height':'40%',
                                                                            'color':'blue',
                                                                            'background-color':'yellow',
                                                                            'font-size':'20px'
                                                                            }
                                                                            )

                                                                            ], style=self.CubicPagePanel_narrow,
                                                                            ),


                                                                            ],   style={'justify-content':'center',
                                                                                'align-items':'center',
                                                                                'display':'inline-block',
                                                                                'margin':'0 auto',
                                                                                'width':'90%',
                                                                                'height':'40%',                                                   
                                                                            }
                                                                            ),
                                                        ],#children      
                                                    ),
                                                md=4,          
                                                style=self.OneThirdPagePanel
                                                    ), 
                            ]
                        ),
                        ],

                    ),

        ####################################################################################################
        ########################################### ENDS ####################################################
            ]),
        ####################################################################################################
        ####################################################################################################
         ]),
        ],
        style={'width': '95%', 
                'display': 'inline-block',
                'align-items': 'center', 
                'padding': '20px',
                'align-items': 'center',
                'margin':'50px',
                'justify-content': 'center', }
        )

####################################################################################################
####################################################################################################
        return layout
     
if __name__=='__main__':
    filename = sys.argv[1]
    NLPToolClassClass = NLPToolClass(filename)
    NLPToolClassClass.app.run_server(debug=False, port=9132, dev_tools_hot_reload=False)