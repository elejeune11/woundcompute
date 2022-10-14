import os
from pathlib import Path
import time
from woundcompute import image_analysis as ia

self_path_file = Path(__file__)
self_path = self_path_file.resolve().parent

folder_files = self_path.joinpath("files").resolve()
folder_dataset = folder_files.joinpath("sample_dataset").resolve()

for idx in range(1, 4):
	input_location = folder_dataset.joinpath("s%i" % (idx))
	try:
		time_all, action_all = ia.run_all(input_location)
		print("tissue number:", idx, "time:", time.time())
	except Exception as ex:
		time_all.append(time.time())
		print("tissue number:", idx, "time:", time.time())
		print("---------ERROR OF SOME DESCRIPTION HAS HAPPENED-------")
		print(ex)
		print("------------------------------------------------------")
