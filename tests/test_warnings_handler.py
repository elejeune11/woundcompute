# def test_save_bg_shift_warning():
#     folder_path = example_path("test_ph1_movie_mini_large_bg_shift")
#     sample_name = "test_sample"
#     large_shift_frame_ind = np.array([1])
#     _, _ = ia.run_all(folder_path)
#     output_p = ia.save_bg_shift_warning(folder_path,sample_name,large_shift_frame_ind)
#     assert output_p.is_file()