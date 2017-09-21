from flask import render_template, make_response
from flaskexample import app
from flask import request
from flask import jsonify

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
    path = 'webapp/'
    sender_name_to_email = pickle.load(open(path + 'sender_name_to_email.p', 'rb'))
    people = sorted(list(sender_name_to_email.keys()))
    
    return render_template('input.html', people = people)


@app.route('/output')
def output():
    select = request.args.get('person')
    
    path = 'webapp/'
    text_features = joblib.load(path + '12-text_features.pkl')
    
    sender_name_to_email = pickle.load(open(path + 'sender_name_to_email.p', 'rb'))
    people = sorted(list(sender_name_to_email.keys()))
    
    name = sender_name_to_email[select]
    col = 'compound'
    
    fig, ax = plt.subplots();
    fig.set_size_inches(8, 6)

    #color individual emails
    indiv = text_features[text_features['sender'] == name]
    indiv_dates = date2num(indiv['date'].tolist())
    ax.plot_date(indiv_dates, np.array(indiv[col]), markersize = 1.5, c = 'blue')

    #plot individual mean
    over_time = text_features.set_index(pd.DatetimeIndex(text_features['date']))
    indiv_mean = over_time[over_time['sender'] == name].resample('M')['compound'].mean()
    plt.plot(indiv_mean, 'b', label = select + ' mean sentiment');
    
    #plot company mean
    mean_sentiment = over_time.resample("M")['compound'].mean()
    plt.plot(mean_sentiment, '--', c = 'dimgray', label = 'company mean sentiment');

    #plot all points
    x_axis_dates = date2num(text_features['date'].tolist())
    ax.plot_date(x_axis_dates, np.array(text_features[col]), markersize = 0.5, c = 'gray');

    plt.gcf().autofmt_xdate();
    ax.set_xlim([dt(1999, 12, 1), dt(2002, 2, 28)]);

    plt.title('Sentiments over time of emails sent by {}'.format(select), fontsize = 15);
    plt.ylabel('Sentiment score', fontsize = 14);
    plt.xlabel('Date (Year-Month)', fontsize = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 13);

    lgd = plt.legend(bbox_to_anchor=(1.6, 0.55), 
                     loc=5, 
                     borderaxespad=0.,
                    fontsize = 14,
                    markerscale = 2);
    
    rand = random.randint(0, 1000)
    ax.figure.savefig('flaskexample/static/image{}.png'.format(rand), 
                      bbox_extra_artists=(lgd,), bbox_inches='tight');
    
    
    #####tsne clusters
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
                size = 7, scatter_kws = {'alpha':0.9, 's':2})
    
    indiv_pts = text_features[text_features['sender'] == name].index
    indiv_pts = tsne_df.iloc[indiv_pts]
    plt.scatter(indiv_pts['dimension 1'], indiv_pts['dimension 2'], 
                c = 'mediumspringgreen', s = 1, label = select);

    plt.title('Types of emails sent by {}'.format(select), fontsize = 15);
    plt.ylabel('dimension 2', fontsize = 14);
    plt.xlabel('dimension 1', fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=13);

    lgd = plt.legend(bbox_to_anchor=(1.8, 0.55), 
                     loc=5, 
                     borderaxespad=0.,
                    fontsize = 14,
                    markerscale = 2.5);

    plt.savefig('flaskexample/static/tsne{}.png'.format(rand), 
                bbox_extra_artists=(lgd,), bbox_inches='tight');
    
    ###individual clusters over time
    cluster_colors = {
        'Business discussions':'goldenrod',
        'Casual and personal':'m',
        'Company announcements':'darkgreen',
        'Direct and to the point':'royalblue',
        'Meetings and interviews':'teal',
        'Updates and miscellaneous commentary':'darkred'
    }
    
    fig, ax = plt.subplots();
    fig.set_size_inches(8, 6)
        
    #plot individual mean
    plt.plot(indiv_mean, 'b', label = select + ' mean sentiment');

    #plot company mean
    plt.plot(mean_sentiment, '--', c = 'dimgray', label = 'company mean sentiment');

    #color by cluster labels
    for label, color in cluster_colors.items():
        pts_to_plot = indiv[indiv['cluster_labels'] == label]
        pts_dates = date2num(pts_to_plot['date'].tolist())
        ax.plot_date(pts_dates, pts_to_plot[col], markersize = 2, c = color, label = label);

    plt.gcf().autofmt_xdate();
    ax.set_xlim([dt(1999, 12, 1), dt(2002, 2, 28)]);

    lgd = plt.legend(bbox_to_anchor=(1.8, 0.55), loc=5, borderaxespad=0., 
                     fontsize = 14,
                     markerscale = 2);

    plt.title('Sentiments and types of emails sent by {}'.format(select), fontsize = 15);
    plt.ylabel('Sentiment score', fontsize = 14);
    plt.xlabel('Date (Year-Month)', fontsize = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 13);

    ax.figure.savefig('flaskexample/static/indiv{}.png'.format(rand), 
                      bbox_extra_artists=(lgd,), bbox_inches='tight');
    
    return render_template("output.html", 
                           sentiments='/static/image{}.png'.format(rand), 
                           tsne = '/static/tsne{}.png'.format(rand), 
                           indiv = '/static/indiv{}.png'.format(rand), 
                           people = people)

