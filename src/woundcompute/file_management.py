import json
import os
import re
import shutil
import yaml
import numpy as np
from pathlib import Path
from typing import List,Tuple,Union,Optional,Dict
from woundcompute import image_analysis as ia


def get_sample_info(folder_path:Path,image_type:str='ph1'):
    """
    Given a path to the sample (e.g., /Path/tissue_ai/s03_A03), return dictionary with the following:
    experiment base name (e.g., tissue), sample name (e.g., s03_A03), after injury sample path,
    before injury path (i.e., folder containing all before injury samples), and information for compiled folders.
    """
    path_parts = folder_path.parts
    is_ai_folder=False
    for ind,part in enumerate(path_parts):
        if "_ai" in part:
            sorted_folder_ind = ind-1
            sample_folder_ind = ind+1
            experiment_base_name = part.replace("_ai","")
            is_ai_folder=True
            break
    
    if is_ai_folder is False: # currently, compiled processing only happens when calling WC on ai folders
        sample_info_dict = {}
    elif sample_folder_ind > len(path_parts)-1: # if there is no sample folder in the path, the path doesn't have any data for processing
        sample_info_dict = {}
    else:
        sample_name = path_parts[sample_folder_ind]
        sorted_folder_parts = path_parts[:sorted_folder_ind+1]
        before_injury_folder_parts = sorted_folder_parts + (experiment_base_name+"_bi",)
        sample_compiled_path_parts=sorted_folder_parts+(f"{experiment_base_name}_compiled",sample_name)
        # img_compiled_path_parts=sorted_folder_parts+(f"{experiment_base_name}_compiled",sample_name,f"{image_type}_images")
        well_id = find_well_id(sample_name)

        before_injury_folder_path = Path(*before_injury_folder_parts)
        bi_sample_path = find_matching_before_injury_folder(before_injury_folder_path,well_id)

        sample_info_dict={
            "experiment_base_name": experiment_base_name,
            "sample_name": sample_name,
            "well_id": well_id,
            "after_injury_sample_folder_path": folder_path,
            "before_injury_folder_path": before_injury_folder_path,
            "before_injury_sample_folder_path": bi_sample_path,
            "compiled_folder_path": Path(*sample_compiled_path_parts),
            # "compiled_image_folder_path": Path(*img_compiled_path_parts),
        }
    
    return sample_info_dict,is_ai_folder


def find_well_id(sample_name:str)->str:
    """Given a string with a sample name, return the well ID. e.g., s03_A03 will return A03"""
    well_pattern = re.compile(r'([A-H]\d{2})\b') # pattern for well numbers. e.g., A01, D09, H12
    well_id_match = well_pattern.search(sample_name)
    well_name = well_id_match.group() if well_id_match is not None else None
    return well_name


def get_all_folders_with_yaml_file(base_path)->List:
    """Given a path, search recursively for all folders containing a .yaml file."""
    yaml_folders = set()
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.yaml'):
                yaml_folders.add(root)
                break
    yaml_folders = list(yaml_folders)
    yaml_folders.sort()
    return yaml_folders


def find_matching_before_injury_folder(bi_folder_path:Path,well_id:str)->Path:
    """Given a folder containing all before injury samples, search the sample that matches sample_name by well ID."""
    all_before_injury_sample_folders = get_all_folders_with_yaml_file(bi_folder_path)
    matching_bi_path = None
    for bi_folder in all_before_injury_sample_folders:
        cur_bi_samp_path = Path(bi_folder)
        cur_bi_samp_name = cur_bi_samp_path.parts[-1]
        if well_id in cur_bi_samp_name:
            matching_bi_path = cur_bi_samp_path
            break
    return matching_bi_path


def rename_file_for_compiling(
    filename,
    time_ind:int=0,
    time_digit_count:int=3,
    experiment_base_name:str=None,
    well_id:str=None,
):
    """Given a file name, rename before compiling according to the time index."""
    if isinstance(filename,Path):
        filename = filename.name

    base, ext = os.path.splitext(filename)

    if experiment_base_name is not None and well_id is not None:
        new_base = f"{experiment_base_name}_compiled_{well_id}"
    else:
        new_base = re.sub(r"_(ai|bi)(?=_|\.|$)","_compiled",base)

    time_pattern = re.compile(r't\d+\b')
    time_searched = time_pattern.search(new_base)
    if time_searched is None:
        new_base = new_base + f"_t{time_ind:0{time_digit_count}d}"
    else:
        time_str = time_searched.group()
        new_base = new_base.replace(time_str,f"t{time_ind:0{time_digit_count}d}")
    
    new_filename = new_base + ext
    return new_filename


def rename_all_files_for_compiling(filename_list,starting_ind:int=0,experiment_base_name:str=None,well_id:str=None):
    """Given a list of file names, rename all files for compiling."""
    new_filename_list = []
    for ind,fn in enumerate(filename_list):
        cur_time_ind = ind+starting_ind
        new_fn = rename_file_for_compiling(fn,cur_time_ind,experiment_base_name=experiment_base_name,well_id=well_id)
        new_filename_list.append(new_fn)
    return new_filename_list


def combine_image_paths(ai_path:Path,matching_bi_path:Path,image_type:str,experiment_base_name:str=None,well_id:str=None):
    """Given paths containing info for before injury and after injury of a sample,
    return the compiled image paths list, new compiled image names, and number of images for each folder."""
    ai_img_folder_path = ai_path / f"{image_type}_images"
    bi_img_folder_path = matching_bi_path / f"{image_type}_images"

    if ai_img_folder_path.is_dir():
        ai_img_paths = ia.image_folder_to_path_list(ai_img_folder_path)
    else:
        raise NotADirectoryError(f"No such directory: '{ai_img_folder_path}'")
        
    if bi_img_folder_path.is_dir():
        bi_img_paths = ia.image_folder_to_path_list(bi_img_folder_path)
    else:
        raise NotADirectoryError(f"No such directory: '{bi_img_folder_path}'")

    all_img_paths = bi_img_paths + ai_img_paths

    num_ai_frames = len(ai_img_paths)
    num_bi_frames = len(bi_img_paths)

    return all_img_paths,num_ai_frames,num_bi_frames


def get_yaml_file_path_in_folder(path):
    list_yaml_file_paths = list(path.glob("**/*.yaml")) + list(path.glob("**/*.yml"))
    yaml_file_path = list_yaml_file_paths[0]
    return yaml_file_path


def read_yaml(yaml_path):
    yaml_path = Path(yaml_path)
    with yaml_path.open('r') as file:
        yaml_data = yaml.safe_load(file) or {}
    return yaml_data


def adjust_ai_low_quality_frame_inds(low_quality_frame_inds,num_bi_frames:int=1):
    bi_inds = np.linspace(0,num_bi_frames-1,num_bi_frames,dtype=int).tolist()
    adjusted_low_quality_frame_inds = bi_inds+[ind+num_bi_frames for ind in low_quality_frame_inds]
    return adjusted_low_quality_frame_inds


def modify_yaml_keys(file_path, new_name, key_changes):
    """
    Rename a YAML file and modify specific keys.
    
    Args:
        file_path (str): Path to the original YAML file.
        new_name (str): New filename for the YAML file.
        key_changes (dict): Dictionary of key changes. The convention is as follow:
            - To change value: {'old_key': new_value}
            - To change key and value: {'old_key': ('new_key', new_value)}
            - To change remove key: {'old_key':'delete_key'}
                           
    """
    path = Path(file_path)
    old_name = path.name
    
    if not path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    with path.open('r') as file:
        data = yaml.safe_load(file) or {}

    for option,option_value in key_changes.items():
        if isinstance(option_value,Tuple):
            new_option_name,new_option_value=option_value
        else:
            new_option_name=option
            new_option_value=option_value
        
        if new_option_value == "delete_key":
            del data[option]
        elif option in data:
            # remove the old key and add the new one with the new value
            del data[option]
            data[new_option_name] = new_option_value
        else:
            data[new_option_name] = new_option_value

    new_path = path.parent / new_name
    with new_path.open('w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

    if old_name != new_name:
        path.unlink()
    
    return new_path


def copy_files(
    source_paths: List[Union[str, Path]],
    destination_dir: Union[str, Path],
    new_filename_list: Optional[List[str]] = None,
    dry_run: bool = False
) -> List[Path]:
    """
    Copy files from source locations to a destination directory with optional name change.
    
    Args:
        source_paths: List of paths to files to copy.
        destination_dir: Directory where files should be copied to.
        new_filename_list: List of new filenames for the destination files.
                          If None, keeps original filenames.
        dry_run: If True, print what would be done without actually copying.
    
    Returns:
        List of destination paths of copied files
    """

    dest_dir = Path(destination_dir)
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    for source_ind, source_path in enumerate(source_paths):
        source = Path(source_path)

        if new_filename_list is not None:
            new_filename = new_filename_list[source_ind]
        else:
            new_filename = source.name

        dest_path = dest_dir / new_filename
        if dry_run:
            print(f"Would copy: {source} -> {dest_path}")
        else:
            shutil.copy2(source, dest_path)  # copy2 preserves metadata
        
        copied_files.append(dest_path)
    
    return copied_files


def create_index_mapping(all_file_paths:List[Path],starting_ind:int=-1):
    len_files = len(all_file_paths)
    inds_map = np.linspace(starting_ind,len_files-2,len_files,dtype=int)
    mapping = {}
    for ii,path in enumerate(all_file_paths):
        mapping[inds_map[ii]] = path.name
    return mapping


def save_dict_as_json(dict_to_save:Dict,filename:str,folder_path:Path):
    base,_ = os.path.splitext(filename)
    full_dict_path = folder_path/f"{base}.json"
    with open(full_dict_path,"w") as f:
        json.dump(dict_to_save,f,indent=2)
    return full_dict_path


def prepare_compiled_folder(sample_folder_path:Path,img_type:str='ph1'):

    # get experiment names, well ID, and paths info
    sample_folder_path = Path(sample_folder_path)
    sample_info_dict,is_ai_folder = get_sample_info(sample_folder_path,img_type)
    if is_ai_folder is True:
        ai_sample_path = sample_folder_path
        sample_info_dict["compiled_folder_path"].mkdir(parents=True, exist_ok=True)

    compiled_img_paths,num_ai_frames,num_bi_frames = combine_image_paths(
        ai_sample_path,sample_info_dict["before_injury_sample_folder_path"],img_type,
        sample_info_dict["experiment_base_name"],sample_info_dict["well_id"]
    )
    
    ai_yaml_file_path = get_yaml_file_path_in_folder(ai_sample_path)
    ai_yaml_data = read_yaml(ai_yaml_file_path)

    adjusted_low_quality_frame_inds = adjust_ai_low_quality_frame_inds(ai_yaml_data["low_quality_frame_inds"],num_bi_frames)

    file_index_mapping=create_index_mapping(compiled_img_paths,-1)

    return compiled_img_paths,sample_info_dict,file_index_mapping,adjusted_low_quality_frame_inds
