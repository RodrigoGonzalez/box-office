import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys, re, math
# import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
import unidecode

def cluster(df, role_type, n):
	columns = ['total_lifetime_earnings', 'year', 'gender', 'total_box_office_revenues']
	X = df[columns].get_values()
	ids = df['name_id'].get_values()

	scaler = StandardScaler().fit(X)
	X_train = scaler.transform(X)

	# Compute clustering with MeanShift

	# The following bandwidth can be automatically detected using
	bandwidth = estimate_bandwidth(X_train, quantile=0.2, n_samples=df.shape[0])

	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
	ms.fit(X_train)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_

	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)

	print("number of estimated clusters : %d" % n_clusters_)

	return labels

def info_transform(data):
	info_type_id = int(data[0])
	string = data[1].replace("\\", "")
	info_type = int(info_type_id)

	if info_type == 21:
		nums = map(int, re.findall('\d+', string))
		if len(nums) == 1:
			return nums
		else:
			year_list = [int(x) for x in nums if len(str(x)) == 4]
			return year_list

	elif info_type == 27:
		salary_info = string.decode('utf-8').encode('ascii', 'replace')
		last_string_1 = salary_info.split('::')[-1]
		power = last_string_1.count(',000')
		if power > 2:
			power = 2

		if '?' in last_string_1:
			last_string_2 = re.sub("[,]", "", last_string_1.split('?')[-1])
			conversion_factor = 1.6
		elif '$' in last_string_1:
			last_string_2 = re.sub("[,]", "", last_string_1.split('$')[-1])
			conversion_factor = 1.0
		else:
			return 0.0

		sub_strings = last_string_2.split()

		if len(sub_strings) == 1:
			sub_string = re.sub(".00", "", str(sub_strings))

			try:
				return int([re.sub('[!@#$/:.,A-Za-z()+%\\;]', '', sub_string[0])]) * conversion_factor
			except:
				return 0.0
		elif len(sub_strings) == 0:
			return 0.0

		amount = [re.sub(".00", "", word) for word in sub_strings]
		try:
			process = amount[0]
			salary = int([re.sub('[!@#$/:.,A-Za-z()+%\\;]', '', process)][0]) * conversion_factor
		except:
			salary = 0.0

		redic = len(str(salary).split(".")[0])

		if redic > 8:
			return 0.0
		else:
			return salary

# Plot Figure
#
# plt.figure(1)
# plt.clf()
#
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()


if __name__ == "__main__":
	n = int(sys.argv[1])
	assert n in [1, 3, 4, 8]

	df_basedata = pd.read_csv('../html_postgres/movie_revs.csv')
	df_basedata['movie_revs_id'] = df_basedata['Unnamed: 0']
	into_df = df_basedata[['movie_revs_id', 'year', 'total_revenues']]
	df_movie_cluster = pd.read_csv('../html_postgres/cluster_data.csv')
	df_movie_cluster.drop_duplicates(subset='md5sum', inplace=True)
	# Join movie revenue info into df
	df = pd.read_csv('../html_postgres/person_cluster_data.csv')


	# Columns to transform

	df['gender'] = df.gender.replace(['m', 'f'], [1, 0])
	info_df = df[['info_type_id', 'person_info']].get_values()
	df['cleaned_info'] = [info_transform(x) for x in info_df]
	df['info_type_id'] = df['info_type_id'].astype(int)


	# Drop uncredited because probably unimportent
	df = df[df['cast_info_note'] != '(uncredited)']
	df.drop('cast_info_note', axis=1, inplace=True)
	df.drop('person_role_id', axis=1, inplace=True)
	df.drop('nr_order', axis=1, inplace=True)

	# Convert to ints
	df['name_id'] = df['name_id'].astype(int)
	df['movie_id'] = df['movie_id'].astype(int)
	df['name_id'] = df['name_id'].astype(int)
	df['role_id'] = df['role_id'].astype(int)
	df['movie_revs_id'] = df['movie_revs_id'].astype(int)


	df = df.dropna(subset = ['gender'])

	df['gender'] = df['gender'].astype(int)

	# Generate person unique info
	birth_date_df = df[df['info_type_id'] == 21]
	birth_date_df.drop_duplicates(subset='name_md5sum', inplace=True)

	salary_df = df[df['info_type_id'] == 27]

	salaries = salary_df.groupby('name_md5sum')['cleaned_info'].sum()
	salaries_df = pd.DataFrame(salaries, columns=['cleaned_info'])
	salaries_df['name_md5sum'] = salaries_df.index

	result_salary = pd.merge(df, salaries_df, on='name_md5sum')

	# save all of this hard work
	result_salary.drop(['cleaned_info_x', 'person_info', 'info_type_id'], axis=1, inplace=True)
    # result_salary.to_csv('../html_postgres/all_people_result_salary.csv', mode = 'w', index=False)

	cluster_df = pd.merge(result_salary, into_df, on='movie_revs_id')
	cluster_df.to_csv('../html_postgres/cluster_data_frame.csv', mode = 'w', index=False)
	cluster_df['life_time_earnings'] = cluster_df['total_revenues']
	cluster_df.drop('total_revenues', axis=1, inplace=True)

	career_revenues = cluster_df.groupby('name_id')['life_time_earnings'].sum()
	career_revenues_df = pd.DataFrame(career_revenues, columns=['life_time_earnings'])
	career_revenues_df['name_id'] = career_revenues_df.index
	cluster_df_2 = pd.merge(cluster_df, career_revenues_df, on='name_id')

	df_actor = cluster_df_2.loc[cluster_df_2['role_id'].isin([1, 2])]
	df_producer = cluster_df_2[cluster_df_2['role_id'] == 3]
	df_writer = cluster_df_2[cluster_df_2['role_id'] == 4]
	df_cine = cluster_df_2[cluster_df_2['role_id'] == 5]
	df_director = cluster_df_2[cluster_df_2['role_id'] == 8]

	if n == 1:
		data = df_actor
		role_type = 'actor'
	elif n == 3:
		data = df_producer
		role_type = 'producer'
	elif n == 4:
		data = df_writer
		role_type = 'writer'
	else:
		data = df_director
		role_type = 'director'

	data['total_box_office_revenues'] = data['life_time_earnings_y']
	data['total_lifetime_earnings'] = data['life_time_earnings_x']
	data.drop(['life_time_earnings_y', 'life_time_earnings_x', 'cleaned_info_y', 'movie_id'], axis=1, inplace=True)

	movie_actor_key = cluster_df_2[['name', 'name_id', 'name_md5sum', 'movie_id', 'movie_revs_id', 'md5sum_movie']]
	movie_actor_key.to_csv('../html_postgres/movie_actor_key.csv', mode = 'w', index=False)

	data_df = data.groupby('name_id')['total_lifetime_earnings', 'year', 'gender', 'total_box_office_revenues'].mean()
	data_df['name_id'] = data_df.index

	data_df.to_csv('../html_postgres/person_clusters_{}_{}.csv'.format(role_type, n), mode = 'w', index=False)

	cluster(data_df, n)
