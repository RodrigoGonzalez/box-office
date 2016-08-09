from selenium import webdriver

path_to_chromedriver = '/Users/rodrigogonzalez/Dropbox/Galvanize/box-office/selenium/chromedriver'
# change path as needed

browser = webdriver.Chrome(executable_path = path_to_chromedriver)

url = 'http://www.metacritic.com/'
browser.get(url)

# For rottentomatoes
# name = 'search'
# id = 'search-term'

browser.find_element_by_id('search_term').clear()
browser.find_element_by_id('search_term').send_keys('Gladiator (2000)')

browser.find_element_by_xpath('//*[@id="header_brand_column"]/div[1]/form/div/div/div[1]/button').click()

//*[@id="movie_results_ul"]/li[1]/div/div/a

dyn_frame = browser.find_element_by_xpath('//*[@id="movie_results_ul"]/li[1]/div/div/a')

framename = dyn_frame.get_attribute('href')

# Go to the movie URL
browser.get(framename)
