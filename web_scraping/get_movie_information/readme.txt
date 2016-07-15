Get Summary and Daily
Summary at url of form:
http://www.boxofficemojo.com/movies/?page=main&id=pixar2015.htm

Daily box office results at url of form:
http://www.boxofficemojo.com/movies/?page=daily&view=chart&id=pixar2015.htm&adjust_yr=2016&p=.htm

The download_boxoffice.py file generates links to scrape

# Run from movie_summaries folder
wget --tries=100 -i ../summary.txt

FINISHED --2016-07-12 21:59:30--
Total wall clock time: 43m 59s
Downloaded: 16238 files, 299M in 2m 10s (2.31 MB/s)


# Run from boxoffice_results folder
wget --tries=100 -i ../boxoffice.txt

FINISHED --2016-07-12 22:24:01--
Total wall clock time: 39m 24s
Downloaded: 16246 files, 374M in 2m 35s (2.42 MB/s)
