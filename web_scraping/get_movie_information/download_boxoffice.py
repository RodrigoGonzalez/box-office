import pandas as pd
import numpy as np

# run to get the boxoffice and summary url's

def make_url_list(url, filname, output_file):
    names = ['movie_name']
    movie_id = pd.read_csv(filename, names=names)

    if len(url) == 1:
        url.append("")

    begin, end = url

    movie_id['url'] = begin + movie_id['movie_name'] + end

    movie_id['url'].to_csv(output_file.__str__() + '.txt', sep='\t', index=False)

    return

if __name__ == '__main__':
    # URL to use
    url_summary = ['http://www.boxofficemojo.com/movies/?page=main&']
    url_boxoffice = ['http://www.boxofficemojo.com/movies/?page=daily&view=chart&', '&adjust_yr=2016&p=.htm']

    # filename to extract keys from
    filename = '../get_movie_names/movie_names.csv'

    # run to get the boxoffice and summary url's

    make_url_list(url_summary, filename, 'summary')
    make_url_list(url_boxoffice, filename, 'boxoffice')
