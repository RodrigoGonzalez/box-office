import psycopg2 as pg
import sys
from time import time

if sys.argv[1:]:
    dbname = sys.argv[1]
else:
    dbname = 'movies'

db = pg.connect("dbname='%s'" % dbname)

def list_tables():
    c = db.cursor()
    c.execute('''
            select table_name
            from information_schema.tables
            where table_schema = 'public'
            ''')

    result = [row[0] for row in c.fetchall()]
    c.close()
    return result

def count(t):
    c = db.cursor()
    c.execute('''
        select count(*) from %s
    ''' % t)
    result = c.fetchone()[0]
    c.close()
    return result

start = time()
print "Tables"
print "------"
total = 0
for t in list_tables():
    c = count(t)
    print "%30s: %d" % (t, c)
    total += c
print "========"
print "Total", total
print "Done in {0:.2f} seconds".format(time() - start)
