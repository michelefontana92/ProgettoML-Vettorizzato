from Utilities.Utility import *


path = "../Results_CSV"

best_file, best_vl_error=analyze_result_csv(path,5)
print("Best_file ",best_file)
print("Best_vl_error ",best_vl_error)