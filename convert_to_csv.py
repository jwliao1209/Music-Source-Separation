import museval

# Load the saved .pandas file
def load_evaluation_results(filename):
    method = museval.MethodStore()
    method.load(filename)
    return method

# Example usage
results_file = '/home/jiawei/Desktop/github/Source-Separation/open-unmix-pytorch/open-unmix.pandas'
method_store = load_evaluation_results(results_file)

eval_results = method_store.df
print(type(eval_results))
eval_results.to_csv("temp.csv", index=False)
