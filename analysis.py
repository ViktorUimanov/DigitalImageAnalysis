
import parsing_data
def complete_analysis(main_company, inf_comp_domains, companies_domains, find_lexemes):
    # data collection
    inf_comp_tokens = parsing_data.sbor_zapisei_vk(inf_comp_domains)

    test = parsing_data.sbor_zapisei_vk(companies_domains)
    test_text = test[0]
    test_posts_id = test[1]
    test_date = test[2]
    test_comp_id = test[3]

    # data analysis
    test_prepr_texts, tokens_len = parsing_data.preprocess_text_in_dict(test_text)
    test_prepr_inf_comps = parsing_data.preprocess_text_in_dict(inf_comp_tokens[0])[0]

    unique_index_results, X_matrix = parsing_data.unique_index(test_prepr_texts)

    test_inf_com_app = parsing_data.inf_comp_appearance(test_prepr_texts, test_prepr_inf_comps)

    test_comms = parsing_data.sbor_kommentov_vk(test_comp_id, test_posts_id, test_date)
    print(test_comms)
    test_prepr_comms, tokens_comms_len = parsing_data.preprocess_text_in_dict(test_comms)

    test_find_lexemes_analyzer = parsing_data.find_lexemes_analyzer(find_lexemes, test_comms, tokens_comms_len)

    test_sentiment = parsing_data.comments_sentiment(test_comms)

    # code for statistics to show bigdata
    token_compont_len_dict = {}

    for key in test_prepr_inf_comps.keys():
        token_component_len = 0
        for texts in test_prepr_inf_comps[key]:
            token_component_len += len(texts.split(' '))
        token_compont_len_dict[key] = token_component_len

    print("Количество собранных токенов информационных компонент:")
    for i_comp_, ic_token in token_compont_len_dict.items():
        print('\t{}: {} токенов'.format(i_comp_, ic_token))
    print("Собрано записей в группах компаний в VK: {}".format(sum([len(test_text[x]) for x in test_text.keys()])))
    print(
        "Собрано комментариев в группах компаний в VK: {}".format(sum([len(test_comms[x]) for x in test_comms.keys()])))

    return {'tokens_len': tokens_len,
            'unique_index_results': unique_index_results,
            'X_matrix': X_matrix,
            'test_inf_com_app': test_inf_com_app,
            'test_find_lexemes_analyzer': test_find_lexemes_analyzer,
            'test_sentiment': test_sentiment}