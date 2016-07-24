import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import codecs
import urllib2
import string
import cPickle as pickle
import re
import time
import csv
from unidecode import unidecode
import js2py
import logging
logging.basicConfig(level=logging.DEBUG)
from urllib2 import Request, urlopen, URLError
from datetime import datetime
import psycopg2
import os
import boto
