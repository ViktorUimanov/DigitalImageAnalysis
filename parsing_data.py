def sbor_zapisei_vk(comp_domains):
    import requests
    import json
    import csv
    from datetime import datetime, date, time, timezone
    from dateutil.relativedelta import relativedelta
    import pandas as pd

    pd.DataFrame(["", ""]).to_excel('All posts.xlsx')

    url = 'https://api.vk.com/method/wall.get'

    # functions to extract year, month and day from unix timestamp date
    def norm_year(date):
        return int(datetime.utcfromtimestamp(date).strftime('%Y'))

    def norm_month(date):
        return int(datetime.utcfromtimestamp(date).strftime('%m'))

    def norm_date(date):
        return str(datetime.utcfromtimestamp(date).strftime('%Y-%m-%d'))

    all_posts_text = {}  # words dictionary
    all_posts_id = {}  # dictionary of posts' id to collect comments
    all_posts_date = {}  # date dictionary for collecting comments
    comp_id = {}

    for comp in comp_domains:

        comp_post_text = []
        comp_post_id = []
        comp_post_date = []

        offset = 0

        today = date.today()

        year_ago = today - relativedelta(years=1)

        dt = datetime.combine(year_ago, time.min)
        year_ago_unix = dt.replace(tzinfo=timezone.utc).timestamp()
        last_post_date = year_ago_unix + 1

        while last_post_date > year_ago_unix:
            inf = requests.get(url, params={
                'access_token': 'e9126258e9126258e9126258ede966938aee912e9126258b6a30a2bbaf2c0a42a0efcc6',
                'v': 5.124,
                'domain': comp_domains[comp],
                'count': 100,
                'offset': offset
                }
                               )
            inf_0 = inf.json()['response']['items']
            c_id = inf_0[0]['owner_id']
            comp_id.update({comp: c_id})
            for post in inf_0:
                post_year = norm_year(post['date'])
                post_month = norm_month(post['date'])

                if post['date'] > year_ago_unix:
                    comp_post_text.append(post['text'])
                    comp_post_id.append(post['id'])
                    comp_post_date.append(post['date'])

            last_post = inf_0[-1]
            last_post_year = norm_year(last_post['date'])
            last_post_date = last_post['date']
            offset += 100

        all_posts_text[comp] = comp_post_text
        all_posts_id[comp] = comp_post_id
        all_posts_date[comp] = comp_post_date

        comp_post_df = pd.DataFrame(comp_post_text)

        print('Собраны посты по компании или компоненте ' + comp)
    return [all_posts_text, all_posts_id, all_posts_date, comp_id]


def sbor_kommentov_vk(comp_id, all_posts_id, all_posts_date):
    import requests
    import json
    import csv
    from datetime import datetime, date, time, timezone
    from dateutil.relativedelta import relativedelta

    import pandas as pd

    pd.DataFrame(["", ""]).to_excel('All comms.xlsx')
    url = 'https://api.vk.com/method/wall.getComments'

    def norm_month(date):
        return int(datetime.utcfromtimestamp(date).strftime('%m'))

    token_1 = '3c53fd1c3c53fd1c3c53fd1c333c27856933c533c53fd1c63e2a32dfabc9354f8c687ab'
    token_2 = 'c698f6bec698f6bec698f6beacc6ec07dccc698c698f6be9929a1afb8f64b7d24c5941f'
    token_used = token_1
    req_counter = 0

    comp_tonality = {}
    all_comms = {}

    for comp in all_posts_id:
        comp_comms = []

        date_id = dict(zip(all_posts_date[comp], all_posts_id[comp]))

        # selection of id-posts by 10 for each month

        selected_date_id = {}

        month = 12
        post_counter = 10
        for post in all_posts_date[comp]:
            if norm_month(post) != month:
                month = norm_month(post)
                post_counter = 10
            if post_counter > 0:
                selected_date_id[post] = date_id[post]
                post_counter -= 1

        for post in selected_date_id.values():
            req_counter += 1
            if req_counter > 1999:
                token_used = token_2
            inf = requests.get(url, params={'access_token': token_used,
                                            'v': 5.124,
                                            'owner_id': comp_id[comp],
                                            'post_id': post,
                                            'count': 100
                                            }
                               )
            inf_0 = inf.json()['response']['items']
            for comm in inf_0:
                if 'text' in comm:
                    text_com = comm['text']
                    comp_comms.append(text_com)

        all_comms[comp] = comp_comms

        comp_comms_df = pd.DataFrame(comp_comms)

    return all_comms


# improved tokenizer
import re
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
import nltk

nltk.download("stopwords")
# --------#
from string import punctuation

from nltk.corpus import stopwords

russian_stopwords = stopwords.words("russian")


def preprocess_text(text):
    words = re.split(r'[^а-яА-Я]', text)  # break text into words
    tokens = list()
    for word in words:
        p = morph.parse(word)[0]
        if p.tag.POS not in ['NPRO', 'PREP', 'CONJ', 'PRCL', 'INTJ', 'NUMR']:
            tokens.append(p.normal_form)
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token != " " \
              and token.strip() not in punctuation]
    company_tokens_text_len = len(tokens)

    return " ".join(tokens), company_tokens_text_len


def preprocess_text_in_dict(dict_):
    prepr_dict = {}
    tokens_len = {}
    for company in dict_.keys():
        prepr_texts_and_len = [preprocess_text(x) for x in dict_[company]]
        prepr_texts = [x[0] for x in prepr_texts_and_len]
        len_ = [x[1] for x in prepr_texts_and_len]
        prepr_dict[company] = prepr_texts
        tokens_len[company] = sum(len_)
    return prepr_dict, tokens_len


def join_text(dict_):
    joined_comms = {}

    for comp, comms in dict_.items():
        joined = ". ".join(comms)
        joined_comms[comp] = joined

    return joined_comms


def unique_index(dict_):
    company_tokens = join_text(dict_)

    from sklearn.feature_extraction.text import CountVectorizer
    import scipy
    import numpy as np

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(company_tokens.values()).toarray()

    rasst_list = []

    for u in X:
        rasst_list_u = []
        for v in X:
            rasst = scipy.spatial.distance.cosine(u, v)
            if rasst != 0:
                rasst_list_u.append(rasst)
        rasst_list.append(np.mean(rasst_list_u))

    image_uniqueness = dict(zip(company_tokens.keys(), rasst_list))

    return image_uniqueness, X


def inf_comp_appearance(companies_tokens, component_tokens):
    inf_component_tokens = join_text(component_tokens)
    company_tokens = join_text(companies_tokens)

    from sklearn.feature_extraction.text import CountVectorizer
    import scipy
    import numpy as np

    vectorizer = CountVectorizer()

    inf_component_tokens.update(company_tokens)
    X = vectorizer.fit_transform(inf_component_tokens.values()).toarray()

    rasst_list = []

    for u in X[:len(component_tokens)]:
        rasst_comp = []
        for v in X[len(component_tokens):]:
            rasst = scipy.spatial.distance.cosine(u, v)
            rasst_comp.append(rasst)
        rasst_companies_dict = dict(zip(companies_tokens.keys(), rasst_comp))
        rasst_list.append(rasst_companies_dict)

    rasst_dict = dict(zip(component_tokens.keys(), rasst_list))

    return rasst_dict


def find_lexemes_analyzer(find_lexemes_dict, test_comms_dict_, tokens_comms_len):
    common_lex_list = []

    for lists_ in find_lexemes_dict.values():
        common_lex_list.extend(lists_)

    lexemes_list = []

    for word in common_lex_list:
        word_lexemes_list = morph.parse(word)[0].lexeme
        for lexeme_elem in range(len(word_lexemes_list)):
            lexemes_list.append(word_lexemes_list[lexeme_elem][0])

    unique_lexemes_list = list(set(lexemes_list))

    import re
    from collections import Counter

    counter_dict = {}
    for label in test_comms_dict_.keys():
        word_list = []
        for line in test_comms_dict_[label]:
            line = line.lower()
            word_list_line = re.split(r'[^а-яА-Я]', line)
            for word in word_list_line:
                if word != '':
                    word_list.append(word)

        c = Counter(word_list)

        needed_dict = {}
        for k, v in c.items():
            if k in unique_lexemes_list:
                needed_dict[k] = v

        sum_ = sum(needed_dict.values())
        try:
            freq = sum_ / tokens_comms_len[label] * 1000
        except:
            freq = 0
        counter_dict[label] = freq

    return counter_dict


def comments_sentiment(comments_dict):
    import numpy as np
    from dostoevsky.tokenization import RegexTokenizer
    from dostoevsky.models import FastTextSocialNetworkModel
    from sklearn.preprocessing import StandardScaler

    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)

    comp_tonality = []
    comp_emotionality = []

    for company, comments in comments_dict.items():
        company_tonality = []
        company_emotionality = []
        for comment in comments:
            if comment != '':
                stata = model.predict([comment])
                tonality = stata[0]['positive'] / stata[0]['negative']
                emotionality = (stata[0]['positive'] + stata[0]['negative']) / stata[0]['neutral']
                if tonality > 100:
                    tonality = 100
                if tonality < 0.01:
                    tonality = 0.01
                if emotionality > 100:
                    emotionality = 100
                if emotionality < 0.01:
                    emotionality = 0.01
                company_tonality.append(tonality)
                company_emotionality.append(emotionality)
        comp_tonality.append(np.mean(company_tonality))
        comp_emotionality.append(np.mean(company_emotionality))

    dict_comp_tonality = dict(zip(comments_dict.keys(), comp_tonality))
    dict_comp_emotionality = dict(zip(comments_dict.keys(), comp_emotionality))

    return dict_comp_tonality, dict_comp_emotionality


def get_color_map(compa_list):
    color_list_ = []
    for company_ in compa_list:
        if company_ != main_company_:
            color_list_.append('b')
        else:
            color_list_.append('r')
    return color_list_


comp_anly = complete_analysis(main_company_, inf_comp_domains_, companies_domains_, find_lexemes_)
tokens_len = comp_anly['tokens_len']
unique_index_results = comp_anly['unique_index_results']
X_matrix = comp_anly['X_matrix']
test_inf_com_app = comp_anly['test_inf_com_app']
test_find_lexemes_analyzer = comp_anly['test_find_lexemes_analyzer']
test_sentiment = comp_anly['test_sentiment']
companies_list = list(companies_domains_.keys())
comp_anly