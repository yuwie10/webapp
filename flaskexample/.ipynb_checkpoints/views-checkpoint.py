from flask import render_template, make_response
from flaskexample import app
from flask import request
from flask import jsonify

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np

from sklearn.externals import joblib
import pickle

import random
from datetime import datetime as dt
from matplotlib.dates import date2num

import matplotlib.pyplot as plt
import seaborn as sns

@app.route('/')
@app.route('/search/', methods=['GET', 'POST'])
#def index():
    #return render_template("index.html")

@app.route('/input', methods=['GET'])
def input():
    
    name_to_email = {
        'Eric Bass':'eric.bass@enron.com',
        'John Arnold':'john.arnold@enron.com',
        'Phillip K. Allen':'phillip.allen@enron.com',
        'Sally Beck':'sally.beck@enron.com',
        'Shona Wilson':'shona.wilson@enron.com'
    }
    
    people = sorted(list(name_to_email.keys()))
    
    return render_template('input.html', people = people)


@app.route('/output')
def output():
    name = request.args.get('person')
    
    path = 'webapp/'
    text_features = joblib.load(path + '12-features_webapp.pkl')
    
    #sender_name_to_email = pickle.load(open(path + 'sender_name_to_email.p', 'rb'))
    #people = sorted(list(sender_name_to_email.keys()))
    
    name_to_email = {
        'Eric Bass':'eric.bass@enron.com',
        'John Arnold':'john.arnold@enron.com',
        'Phillip K. Allen':'phillip.allen@enron.com',
        'Sally Beck':'sally.beck@enron.com',
        'Shona Wilson':'shona.wilson@enron.com'
    }
    
    people = sorted(list(name_to_email.keys()))
    
    email = name_to_email[name]
    col = 'compound'
    
    fig, ax = plt.subplots();
    fig.set_size_inches(11, 9)
    
    #create necessary dfs, etc. for plots
    over_time = text_features.set_index(pd.DatetimeIndex(text_features['date']))
    per_month = over_time.resample('M')
    mean_sentiment = per_month['compound'].mean()
    std = per_month['compound'].std()
    x_axis_dates = date2num(text_features['date'].tolist())
    x_fill = date2num(mean_sentiment.index.tolist())
    y_label = ('Sentiment \n' + r'$\longleftarrow$' + 
               ' negative   ' + r'$\longleftrightarrow$' + '   neutral   ' +
               r'$\longleftrightarrow$' + '   positive ' + 
               r'$\longrightarrow$')

    #create necessary dfs, etc. for plotting of indiv info
    indiv = text_features[text_features['sender'] == email]
    indiv_dates = date2num(indiv['date'].tolist())
    indiv_per_month = over_time[over_time['sender'] == email].resample('M')
    indiv_mean = indiv_per_month['compound'].mean()
    indiv_std = indiv_per_month['compound'].std()
    x_indiv_fill = date2num(indiv_mean.index.tolist())

    #color individual emails on plot
    ax.plot_date(indiv_dates, np.array(indiv[col]), 
                 markersize = 1, c = 'blue', marker = '.')

    #plot individual mean
    plt.plot(indiv_mean, 'b', label = name + ' mean sentiment', linewidth = 3);
    plt.fill_between(x_indiv_fill, indiv_mean - indiv_std, indiv_mean + indiv_std,
                    alpha = 0.2, facecolor = 'cornflowerblue');

    #plot all points
    x_axis_dates = date2num(text_features['date'].tolist())
    ax.plot_date(x_axis_dates, np.array(text_features[col]), 
                 markersize = 0.5, c = 'gray', marker = '.');

    #plot company mean and std
    plt.plot(mean_sentiment, '--', c = 'dimgray', 
             label = 'company mean sentiment', linewidth = 3);
    plt.fill_between(x_fill, mean_sentiment - std, mean_sentiment + std,
                    alpha = 0.2, facecolor = 'gray');

    plt.gcf().autofmt_xdate();
    ax.set_xlim([dt(1999, 12, 20), dt(2002, 2, 28)]);
    ax.set_ylim(-0.5, 0.5);

    plt.title('Sentiments over time of emails sent by {}'.format(name), fontsize = 18, y = 1.1);
    plt.xlabel('Date (Year-Quarter)', fontsize = 17, labelpad = 15);
    plt.ylabel(y_label, fontsize = 17, labelpad = 20);

    #axis ticks
    plt.tick_params(axis = 'both', which = 'major', 
                    labelsize = 13, labelleft = 'off');

    x_pos = ax.get_xticks()
    x_labels = np.array(['2000-Q1', '2000-Q2', '2000-Q3', '2000-Q4', 
                        '2001-Q1', '2001-Q2', '2001-Q3', '2001-Q4', 
                        '2002-Q1'])
    plt.xticks(x_pos, x_labels)

    lgd = plt.legend(bbox_to_anchor = (0.99, 0.02), 
                     loc = 'lower right', 
                     borderaxespad = 0.,
                     fontsize = 15,
                     markerscale = 2);
    
    rand = random.randint(0, 100000)
    ax.figure.savefig('flaskexample/static/image{}.png'.format(rand), 
                      bbox_extra_artists=(lgd,), bbox_inches='tight');
    
    
    
    #####tsne clusters######
    
    tsne_df = joblib.load(path + 'tsne_df.pkl')
    colors = [
        'darkred',
        'royalblue',
        'goldenrod',
        'm',
        'darkgreen',
        'teal'
    ]
    
    g = sns.lmplot('dimension 1', 'dimension 2', tsne_df,
                hue = 'cluster_labels', legend = False,
                fit_reg = False, palette = colors, 
                size = 9, scatter_kws = {'alpha':0.9, 's':2})

    #plot individual's emails
    indiv_pts = text_features[text_features['sender'] == email].index
    indiv_pts = tsne_df.iloc[indiv_pts]
    plt.scatter(indiv_pts['dimension 1'], indiv_pts['dimension 2'], 
                c = 'mediumspringgreen', s = 2, label = name);

    plt.title('Types of emails sent by {}'.format(name), 
              fontsize = 20, y = 0.97);

    plt.axis('off');

    lgd = plt.legend(loc = 'lower center',
                    ncol = 2,
                    borderaxespad = 0.,
                    fontsize = 14,
                    markerscale = 2.4,
                    handletextpad = 0.2)

    lgd.get_frame().set_alpha(1);

    plt.savefig('flaskexample/static/tsne{}.png'.format(rand), 
                bbox_extra_artists = (lgd,), bbox_inches = 'tight');
    
    
    return render_template('output.html', 
                           sentiments='/static/image{}.png'.format(rand), 
                           tsne = '/static/tsne{}.png'.format(rand), 
                           people = people)

