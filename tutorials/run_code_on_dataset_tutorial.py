import os
from pathlib import Path
import time
from woundcompute import image_analysis as ia

self_path_file = Path(__file__)
self_path = self_path_file.resolve().parent

folder_files = self_path.joinpath("files").resolve()
folder_dataset = folder_files.joinpath("sample_dataset").resolve()
sample_names = ["s18_B08","s21_B05","s22_B04"]

for idx,sample_n in enumerate(sample_names):
	start_time = time.time()
	input_location = folder_dataset.joinpath(sample_n)
	try:
		time_all, action_all = ia.run_all(input_location)
		print("tissue number:", idx, ", time:", time.time()-start_time,"s")
	except Exception as ex:
		# time_all.append(time.time())
		print("tissue number:", idx, ", time:", time.time()-start_time,"s")
		print("---------ERROR OF SOME DESCRIPTION HAS HAPPENED-------")
		print(ex)
		print("------------------------------------------------------")
