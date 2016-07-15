from selenium import webdriver

path_to_chromedriver = '/Users/rodrigogonzalez/Dropbox/Galvanize/web_scraping/chromedriver' # change path as needed
browser = webdriver.Chrome(executable_path = path_to_chromedriver)

url = 'http://www.boxofficemojo.com/daily/chart/?view=1day&sortdate=2016-06-13&p=.htm'
browser.get(url)

# Switch into needed frame
browser.switch_to_frame("sis_pixel_sitewide")
browser.find_element_by_id('container')
