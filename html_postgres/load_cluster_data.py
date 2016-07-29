import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
import sys

def connect():
	#Define our connection string
	conn_string = "host='localhost' dbname='movies' user='postgres' password='secret'"

	# print the connection string we will use to connect
	print "Connecting to database\n	->%s" % (conn_string)

	# get a connection, if a connect cannot be made an exception will be raised here
	conn = psycopg2.connect(conn_string)

	# conn.cursor will return a cursor object, you can use this cursor to perform queries
	cursor = conn.cursor()
	print "Connected!\n"
	return conn, cursor

def execute_find_movie(title, year):
	# execute our Query
	try:
		cursor.execute("SELECT * FROM title WHERE title like title AND production_year = year AND kind_id = 1 LIMIT 15")
		records = cursor.fetchall()
	except:
    	conn.rollback()
		print "error: " + title
		records = None
	return records

def prepare_title(title):

	return new_title

def first_pass():
	pass

if __name__ == "__main__":
	conn, cursor = connect()
	df = pd.read_csv('search_titles.csv')



where movetitle like '%10,000%'

cursor.execute("SELECT * FROM title WHERE title = 'The Powerpuff Girls Movie' AND production_year = 2013 AND kind_id = 1 LIMIT 15")
