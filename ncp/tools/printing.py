import numpy as np


def pretty_print_results(results) -> None:
    for res in results:
        if res[1] != {}:
            print("-" * 5, res[0], "-" * 5)
            print("Max epochs ", res[1]["epochs"][0].max())
            get_mean_metric = lambda name: [
                round(m, 3) for m in list(np.stack(res[1][name]).mean(axis=0))
            ]
            mean_test_likelihood = get_mean_metric("test_likelihoods")
            mean_test_rmse = get_mean_metric("test_distances")
            print("** Test likelihoods **", mean_test_likelihood)
            print("** Test RMSE **", mean_test_rmse)
            # Backward compatibility
            if "test_variances" in res[1]:
                mean_test_rmse_variances = get_mean_metric("test_variances")
                print("** Test RMSE var **", mean_test_rmse_variances)
            if "test_samples_distances" in res[1]:
                mean_test_rmse_samples_rmse = get_mean_metric("test_samples_distances")
                print("** Test RMSE samples **", mean_test_rmse_samples_rmse)
            print("\n")
