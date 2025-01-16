import csv
import math
import main
import os
import statistics
import pandas as pd
import scipy.stats as stats


def get_values(tests: int, batch_size: int, activ_func: str, learning_rate: str, path_to_test_folders: str):
    results = []
    for test in range(tests):
        for batch_num in range(batch_size):
            path_to_tests = os.path.join(path_to_test_folders, f"{test+1}_{activ_func}_{learning_rate}")
            file_str = os.path.join(path_to_tests, f"results_{batch_num+1}.csv")
            file_data = pd.read_csv(file_str)
            accuracy = file_data.iloc[-1]["Accuracy"]
            results.append(accuracy)
    return results


def confidence_interval(mean, std_dev, n, confidence=0.95):
    t_value = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    margin_of_error = t_value * (std_dev / math.sqrt(n))
    lower, upper = mean - margin_of_error, mean + margin_of_error
    return lower, upper


def paired_ttest(mean1, mean2, std1, std2, n1, n2):
    means_difference = mean1 - mean2
    lower_delta = math.sqrt((n1*std1**2 + n2*std2**2) / ((n1-1) + (n2-1))) * math.sqrt((1/n1) + (1/n2))
    result = means_difference/lower_delta
    return result


if __name__ == "__main__":
    activ_func_names = ["Tanh", "Sigmoid", "ReLU"]
    learning_rates_str = ["001", "0001", "00001"]
    learning_rates = [0.001, 0.0001, 0.00001]
    activ_funcs = ["tanh", "sigmoid", "relu"]
    num_tests = 5
    batch_size = 10
    ttest_baseline_value = float(1/6)
    n = num_tests
    overall_n = num_tests * len(activ_funcs) * len(learning_rates_str)
    path = os.path.join(os.getcwd(), "tests")

    main.create_folder("results")
    accuracy_file = open(os.path.join(os.getcwd(), "results\\accuracy_results.csv"), mode="w", newline="")
    std_file = open(os.path.join(os.getcwd(), "results\\std_results.csv"), mode="w", newline="")
    ttest_rand_baseline_file = open(os.path.join(os.getcwd(), "results\\ttest_rand_baseline_results.csv"), mode="w", newline="")
    ttest_mean_baseline_file = open(os.path.join(os.getcwd(), "results\\ttest_mean_baseline_results.csv"), mode="w", newline="")
    ci_file = open(os.path.join(os.getcwd(), "results\\confidence_intervals.csv"), mode="w", newline="")
    pval_rand_file = open(os.path.join(os.getcwd(), "results\\p_values_rand_baseline.csv"), mode="w", newline="")
    pval_mean_file = open(os.path.join(os.getcwd(), "results\\p_values_mean_baseline.csv"), mode="w", newline="")
    results_file = open(os.path.join(os.getcwd(), "results\\results.csv"), mode="w", newline="")

    accuracy_writer = csv.writer(accuracy_file)
    std_writer = csv.writer(std_file)
    ttest_rand_baseline_writer = csv.writer(ttest_rand_baseline_file)
    ttest_mean_baseline_writer = csv.writer(ttest_mean_baseline_file)
    ci_writer = csv.writer(ci_file)
    pval_rand_writer = csv.writer(pval_rand_file)
    pval_mean_writer = csv.writer(pval_mean_file)
    results_writer = csv.writer(results_file)

    headers = ["Activation Function \\ Learning Rate", ".001", ".0001", ".00001"]
    accuracy_writer.writerow(headers)
    std_writer.writerow(headers)
    ttest_rand_baseline_writer.writerow(headers)
    ttest_mean_baseline_writer.writerow(headers)
    ci_writer.writerow(headers)
    pval_rand_writer.writerow(headers)
    pval_mean_writer.writerow(headers)

    headers_results = ["Function_LearningRate", "Accuracy", "Lower Bound", "Upper Bound", "Lower Error", "Upper Error", "Standard Deviation", "Random T-test"]
    for func in activ_func_names:
        for lr in learning_rates:
            headers_results.append(f"{func} {lr} Paired T-test")
    results_writer.writerow(headers_results)

    # Get accuracy, std values, ttest random baseline values
    accuracy_list = []
    std_list = []
    ttest_rand_base_list = []
    for i in range(len(activ_funcs)):
        for lr in learning_rates_str:
            values = get_values(num_tests, batch_size, activ_funcs[i], lr, path)
            accuracy_list.append(sum(values) / len(values))
            std_list.append(statistics.stdev(values))
            ttest_rand_base_list.append(((sum(values) / len(values) - ttest_baseline_value) / (statistics.stdev(values) / math.sqrt(n))))

    # get ttest mean of models baseline values
    models_mean = sum(accuracy_list) / len(accuracy_list)
    ttest_mean_base_list = []
    for i in range(len(accuracy_list)):
        ttest_mean_base_list.append((accuracy_list[i] - models_mean) / (std_list[i] / math.sqrt(overall_n)))

    # Calculate and store confidence intervals
    confidence_intervals = []
    for i in range(len(accuracy_list)):
        ci_lower, ci_upper = confidence_interval(accuracy_list[i], std_list[i], n)
        confidence_intervals.append((ci_lower, ci_upper))

    # p-values for each t-test
    p_values_rand = []
    p_values_mean = []
    for t_value in ttest_rand_base_list:
        p_value = 2 * (1 - stats.t.cdf(abs(t_value), df=n - 1))
        p_values_rand.append(p_value)

    for t_value in ttest_mean_base_list:
        p_value = 2 * (1 - stats.t.cdf(abs(t_value), df=overall_n - 1))
        p_values_mean.append(p_value)

    # Write to files
    count = 0
    for func in activ_func_names:
        accuracy_writer.writerow([f"{func}", f"{accuracy_list[count]}", f"{accuracy_list[count+1]}", f"{accuracy_list[count+2]}"])
        std_writer.writerow([f"{func}", f"{std_list[count]}", f"{std_list[count+1]}", f"{std_list[count+2]}"])
        ttest_rand_baseline_writer.writerow([f"{func}", f"{ttest_rand_base_list[count]}", f"{ttest_rand_base_list[count+1]}", f"{ttest_rand_base_list[count+2]}"])
        ttest_mean_baseline_writer.writerow([f"{func}", f"{ttest_mean_base_list[count]}", f"{ttest_mean_base_list[count+1]}", f"{ttest_mean_base_list[count]+2}"])
        ci_writer.writerow([f"{func}", f"{confidence_intervals[count]}", f"{confidence_intervals[count + 1]}", f"{confidence_intervals[count + 2]}"])
        pval_rand_writer.writerow([f"{func}", f"{p_values_rand[count]}", f"{p_values_rand[count + 1]}", f"{p_values_rand[count + 2]}"])
        pval_mean_writer.writerow([f"{func}", f"{p_values_mean[count]}", f"{p_values_mean[count + 1]}", f"{p_values_mean[count + 2]}"])
        count += 3

    # Write to results file
    # ["Function_LearningRate", "Accuracy", "Lower Bound", "Upper Bound", "Lower Error", "Upper Error", "Standard Deviation", "Random T-test", "Paired T-test"]
    count = 0
    for func in activ_func_names:
        for lr in learning_rates:
            low_error = accuracy_list[count] - confidence_intervals[count][0]
            high_error = confidence_intervals[count][1] - accuracy_list[count]
            row_values = [f"{func} {lr}", f"{accuracy_list[count]}", f"{confidence_intervals[count][0]}", f"{confidence_intervals[count][1]}", f"{low_error}", f"{high_error}", f"{std_list[count]}", f"{ttest_rand_base_list[count]}"]
            for index, value in enumerate(accuracy_list):
                row_values.append(paired_ttest(value, accuracy_list[count], std_list[index], std_list[count], n, n))
            results_writer.writerow(row_values)
            count += 1

    accuracy_file.close()
    std_file.close()
    ttest_rand_baseline_file.close()
    ttest_mean_baseline_file.close()
    ci_file.close()
    pval_rand_file.close()
    pval_mean_file.close()
    results_file.close()
