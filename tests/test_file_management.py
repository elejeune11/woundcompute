import glob
import numpy as np
import os
import pytest
import yaml
from pathlib import Path
from woundcompute import file_management as fm
from woundcompute import image_analysis as ia

def files_path():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    return data_path


def example_path(example_name):
    data_path = files_path()
    example_path = data_path.joinpath(example_name).resolve()
    return example_path


def glob_ph1(example_name):
    folder_path = example_path(example_name)
    ph1_path = folder_path.joinpath("ph1_images").resolve()
    name_list = glob.glob(str(ph1_path) + '/*.TIF')
    name_list.sort()
    name_list_path = []
    for name in name_list:
        name_list_path.append(Path(name))
    return name_list


def glob_ph1_sample_folder(example_name,sample_name):
    folder_path = example_path(example_name)
    sample_path = folder_path.joinpath(sample_name).resolve()
    return sample_path


def test_get_sample_info():
    sample_path = glob_ph1_sample_folder("test_for_compiled_folder_ai","sample1_A03")
    samp_info_dict,is_ai_folder = fm.get_sample_info(sample_path)
    assert is_ai_folder is True
    assert samp_info_dict["experiment_base_name"] == "test_for_compiled_folder"
    assert samp_info_dict["sample_name"] == "sample1_A03"
    assert samp_info_dict["well_id"] == "A03"
    assert samp_info_dict["after_injury_sample_folder_path"].name == "sample1_A03"
    assert samp_info_dict["before_injury_folder_path"].name == "test_for_compiled_folder_bi"
    assert samp_info_dict["before_injury_sample_folder_path"].name == "sample2_A03"
    assert isinstance(samp_info_dict["compiled_folder_path"],Path)

    sample_path = glob_ph1_sample_folder("test_for_compiled_folder_bi","sample2_A03")
    samp_info_dict,is_ai_folder = fm.get_sample_info(sample_path)
    assert is_ai_folder is False
    assert samp_info_dict == {}
    
    after_injury_path = example_path("test_for_compiled_folder_ai")
    samp_info_dict,is_ai_folder = fm.get_sample_info(after_injury_path)
    assert is_ai_folder is True
    assert samp_info_dict == {}


def test_find_well_id():
    sample_name = "s95_H08"
    well_id = fm.find_well_id(sample_name)
    assert well_id == "H08"

    sample_name = "s01_Drift"
    well_id = fm.find_well_id(sample_name)
    assert well_id is None

    sample_name = "s100_H133"
    well_id = fm.find_well_id(sample_name)
    assert well_id is None

    sample_name = "s103_J10"
    well_id = fm.find_well_id(sample_name)
    assert well_id is None

def test_get_all_folders_with_yaml_file():
    before_injury_path = example_path("test_for_compiled_folder_bi")
    folders_with_yaml = fm.get_all_folders_with_yaml_file(before_injury_path)
    assert len(folders_with_yaml) == 1

    bad_path = example_path("test_single_fail")
    empty_folder_list = fm.get_all_folders_with_yaml_file(bad_path)
    assert empty_folder_list == []


def test_find_matching_before_injury_folder():
    after_injury_sample_path = glob_ph1_sample_folder("test_for_compiled_folder_ai","sample1_A03")
    samp_info_dict,_ = fm.get_sample_info(after_injury_sample_path)
    before_injury_sample_path = samp_info_dict["before_injury_folder_path"]
    matching_bi_sample_path = fm.find_matching_before_injury_folder(
        before_injury_sample_path,samp_info_dict["well_id"]
        )
    assert isinstance(matching_bi_sample_path,Path)
    assert str(after_injury_sample_path).replace("_ai","")[-3:] == str(matching_bi_sample_path).replace("_bi","")[-3:]


def test_rename_file_for_compiling():
    filename = "tissue_ai_s20_B08_t002.tif"
    new_filename = fm.rename_file_for_compiling(filename,3)
    assert new_filename == "tissue_compiled_s20_B08_t003.tif"

    filename = "tissue_bi_s20_B08.TIFF"
    new_filename = fm.rename_file_for_compiling(filename,0)
    assert new_filename == "tissue_compiled_s20_B08_t000.TIFF"

    filename = "tissue_bs_s20_B08.tiff"
    new_filename = fm.rename_file_for_compiling(filename,10)
    assert new_filename == "tissue_bs_s20_B08_t010.tiff"

    filename = "tissue_ai_s20_B08_t008.TIF"
    new_filename = fm.rename_file_for_compiling(filename,5,experiment_base_name="tissue",well_id="B08")
    assert new_filename == "tissue_compiled_B08_t005.TIF"


def test_rename_all_files_for_compiling():
    filename_list=[
        "tissue_bi_s20_B08.TIFF",
        "tissue_ai_s20_B08_t002.tif",
        "tissue_ai_s20_B08_t004.tiff",
    ]
    new_filename_list = fm.rename_all_files_for_compiling(filename_list,10)
    assert new_filename_list[0] == "tissue_compiled_s20_B08_t010.TIFF"
    assert new_filename_list[1] == "tissue_compiled_s20_B08_t011.tif"
    assert new_filename_list[2] == "tissue_compiled_s20_B08_t012.tiff"


def test_combine_image_paths():
    after_injury_sample_path = glob_ph1_sample_folder("test_for_compiled_folder_ai","sample1_A03")
    samp_info_dict,_ = fm.get_sample_info(after_injury_sample_path)
    before_injury_sample_path = samp_info_dict["before_injury_folder_path"]
    matching_bi_sample_path = fm.find_matching_before_injury_folder(
        before_injury_sample_path,samp_info_dict["well_id"]
        )
    all_img_paths,num_ai_frames,num_bi_frames=fm.combine_image_paths(
        after_injury_sample_path,matching_bi_sample_path,'ph1',
        experiment_base_name=samp_info_dict["experiment_base_name"],
        well_id=samp_info_dict["well_id"],
    )
    assert len(all_img_paths) == 4
    assert num_ai_frames == 3
    assert num_bi_frames == 1
    for img_p in all_img_paths:
        assert img_p.exists()

    with pytest.raises(NotADirectoryError):
        fm.combine_image_paths(Path("/test/test/test"),matching_bi_sample_path,'ph1')

    with pytest.raises(NotADirectoryError):
        fm.combine_image_paths(after_injury_sample_path,Path("/test/test/test"),'ph1')


def test_get_yaml_file_path_in_folder():
    after_injury_sample_path = glob_ph1_sample_folder("test_for_compiled_folder_ai","sample1_A03")
    yaml_file_path = fm.get_yaml_file_path_in_folder(after_injury_sample_path)
    _,ext = os.path.splitext(yaml_file_path.name)
    assert ext == '.yaml'
    assert isinstance(yaml_file_path,Path)


def test_read_yaml():
    after_injury_sample_path = glob_ph1_sample_folder("test_for_compiled_folder_ai","sample1_A03")
    yaml_file_path = fm.get_yaml_file_path_in_folder(after_injury_sample_path)
    yaml_data = fm.read_yaml(yaml_file_path)
    assert yaml_data != {}


def test_adjust_ai_low_quality_frame_inds():
    after_injury_sample_path = glob_ph1_sample_folder("test_for_compiled_folder_ai","sample1_A03")
    yaml_file_path = fm.get_yaml_file_path_in_folder(after_injury_sample_path)
    yaml_data = fm.read_yaml(yaml_file_path)
    adjusted_inds = fm.adjust_ai_low_quality_frame_inds(yaml_data["low_quality_frame_inds"],1)
    assert np.allclose(adjusted_inds,[0,1,2])


def test_modify_yaml_keys():
    after_injury_sample_path = glob_ph1_sample_folder("test_for_compiled_folder_bi","sample2_A03")
    yaml_file_path = fm.get_yaml_file_path_in_folder(after_injury_sample_path)
    key_changes = {
        "low_quality_frame_inds": ("testing_inds",[10,11]),
        "run_before_injury_and_after_injury_together": False,
        "random_new_key": True,
    }
    old_name=yaml_file_path.name
    new_name = "test_name.yaml"
    new_yaml_path = fm.modify_yaml_keys(yaml_file_path,new_name,key_changes)
    with new_yaml_path.open('r') as file:
        new_data = yaml.safe_load(file) or {}
    assert new_data["testing_inds"] == [10,11]
    assert new_data["run_before_injury_and_after_injury_together"] is False
    assert "random_new_key" in new_data.keys()
    assert "low_quality_frame_inds" not in new_data.keys()
    assert new_yaml_path.name == new_name

    reverse_key_changes = {
        "testing_inds": ("low_quality_frame_inds", []),
        "run_before_injury_and_after_injury_together":True,
        "random_new_key":"delete_key",
    }
    old_yaml_path = fm.modify_yaml_keys(new_yaml_path,old_name,reverse_key_changes)
    with old_yaml_path.open('r') as file:
        old_data = yaml.safe_load(file) or {}
    assert old_data["low_quality_frame_inds"] == []
    assert old_data["run_before_injury_and_after_injury_together"] is True
    assert "random_new_key" not in old_data.keys()
    assert "testing_inds" not in old_data.keys()
    assert old_yaml_path.name == old_name

    with pytest.raises(FileNotFoundError):
        fm.modify_yaml_keys(Path('/test/test/test'),"filler",{})


def test_copy_files():
    file = glob_ph1("test_single")[0]
    test_path=Path("test_copy")
    copied_files = fm.copy_files([file],test_path,["new_name.tif"],dry_run=True)
    assert copied_files[0].name == "new_name.tif"

    copied_files = fm.copy_files([file],test_path,dry_run=False)
    assert copied_files[0].exists()


def test_create_index_mapping():
    path_list = [
        Path("/test/file1.tif"),
        Path("/test/file2.tif"),
        Path("/test/file3.tif")
    ]
    known_mapping = {
        -1:"file1.tif",
        0:"file2.tif",
        1:"file3.tif",
    }
    found_mapping = fm.create_index_mapping(path_list,-1)
    assert found_mapping == known_mapping


def test_save_dict_as_json():
    test_path = example_path("test_for_compiled_folder_bi")
    test_dict={"test1":"string1","test2":"string2",}
    result_path = fm.save_dict_as_json(test_dict,"test_json",test_path)
    assert result_path.is_file()


def test_prepare_compiled_folder():
    after_injury_sample_path = glob_ph1_sample_folder("test_for_compiled_folder_ai","sample1_A03")
    compiled_img_paths,sample_info_dict,file_index_mapping,adjusted_low_quality_frame_inds = fm.prepare_compiled_folder(after_injury_sample_path,'ph1')
    known_file_index_mapping = {
        -1:"tissue_bi_s004_A03.TIF",
        0:"tissue_ai_s004_A03_t001.TIF",
        1:"tissue_ai_s004_A03_t002.TIF",
        2:"tissue_ai_s004_A03_t003.TIF",
    }
    assert len(compiled_img_paths) == 4
    assert sample_info_dict["compiled_folder_path"].is_dir()
    assert file_index_mapping == known_file_index_mapping
    assert np.allclose(adjusted_low_quality_frame_inds,[0,1,2])
