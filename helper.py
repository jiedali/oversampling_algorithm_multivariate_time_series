import statistics

def get_mean_max_f1score(f1_score_list):

	mean_f1 = statistics.mean(f1_score_list)
	max_f1 = max(f1_score_list)

	return mean_f1, max_f1