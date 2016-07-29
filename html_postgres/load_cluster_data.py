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

def execute_find_movie(title, year):
	# execute our Query
	cursor.execute("SELECT * FROM title WHERE title = title AND production_year = year LIMIT 15")
	records = cursor.fetchall()
	return records

def prepare_title(title):

	return new_title

if __name__ == "__main__":
	connect()
	df = pd.read_csv('movie_revs.csv')

where movetitle like '%10,000%'
