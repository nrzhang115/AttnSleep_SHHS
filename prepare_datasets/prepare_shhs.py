import os
import numpy as np

import argparse
import glob
import math
import ntpath

import shutil
import urllib
# import urllib2

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from mne.io import concatenate_raws, read_raw_edf
import dhedfreader
import xml.etree.ElementTree as ET

###############################
EPOCH_SEC_SIZE = 30

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/srv/scratch/speechdata/sleep_data/SHHS/edfs/shhs1",
                        help="File path to the PSG files.")
    parser.add_argument("--ann_dir", type=str, default="/home/z5298768/SHHS_annotations",
                        help="File path to the annotation files.")
    parser.add_argument("--output_dir", type=str, default="/srv/scratch/z5298768/SHHS_data",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG C4-A1",
                        help="The selected channel")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        
    # Get selected shhs file
    ids = pd.read_csv("selected_shhs1_files.txt", header=None, names=['a'])
    ids = ids['a'].values.tolist()

    edf_fnames = [os.path.join(args.data_dir, i + ".edf") for i in ids]
    ann_fnames = [os.path.join(args.ann_dir,  i + "-profusion.xml") for i in ids]
    
    # # Get all EDF and XML files from the directories
    # edf_fnames = glob.glob(os.path.join(args.data_dir, "*.edf"))
    # ann_fnames = glob.glob(os.path.join(args.ann_dir, "*.xml"))

    edf_fnames.sort()
    ann_fnames.sort()

    edf_fnames = np.asarray(edf_fnames)
    ann_fnames = np.asarray(ann_fnames)

    apnea_events = [ # based on the annotation xml file
        "Central Apnea", 
        "Obstructive Apnea", 
        "Mixed Apnea", 
        "Hypopnea", 
        "Obstructive Hypopnea", 
        "Central Hypopnea", 
        "Mixed Hypopnea"
    ]

    for file_id in range(len(edf_fnames)):
        if os.path.exists(os.path.join(args.output_dir, edf_fnames[file_id].split('/')[-1])[:-4]+".npz"):
            continue
        
        # Check if annotation file exists
        if not os.path.exists(ann_fnames[file_id]):
            print(f"Annotation file not found: {ann_fnames[file_id]}, skipping corresponding EDF file: {edf_fnames[file_id]}")
            continue
        
        try: 
            print(edf_fnames[file_id])

            raw = read_raw_edf(edf_fnames[file_id], preload=True, stim_channel=None, verbose=None)
            sampling_rate = raw.info['sfreq']
            ch_type = args.select_ch.split(" ")[0]
            select_ch = [s for s in raw.info["ch_names"] if ch_type in s][0]

            # raw_ch_df = raw.to_data_frame(scaling_time=sampling_rate)[select_ch]
            raw_ch_df = raw.to_data_frame(time_format=None)[select_ch] * (1.0 / sampling_rate)
            raw_ch_df = raw_ch_df.to_frame()
            raw_ch_df.set_index(np.arange(len(raw_ch_df)))

            labels = np.ones(len(raw_ch_df))  # Initialize labels as one (for non-apnea events and apnea events are shorter than 10s)
            
            # Read annotation and its header
            t = ET.parse(ann_fnames[file_id])
            r = t.getroot()
            faulty_File = 0
            for event in r.findall('.//ScoredEvent'):
                event_name = event.find('Name').text
                duration_check = float(event.find('Duration').text)
                # Check the duration of apnea events
                if event_name in apnea_events and duration_check >= 10:
                    # Convert time durations to sample indices
                    start_time = float(event.find('Start').text) * sampling_rate
                    duration = duration_check * sampling_rate
                    start_idx = int(start_time)
                    end_idx = int(start_time + duration)
                    labels[start_idx:end_idx] = 0  # Only marking apnea events with 0
            
            raw_ch = raw_ch_df.values

            # Verify that we can split into 30-s epochs
            if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
                raise Exception("Something wrong")
            n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

            # Get epochs and their corresponding labels
            x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
            y = np.asarray(np.split(labels, n_epochs)).astype(np.int32)
            
            assert len(x) == len(y)

            # Select on sleep periods
            w_edge_mins = 30
            nw_idx = np.where(y.sum(axis=1) != 0)[0]
            start_idx = nw_idx[0] - (w_edge_mins * 2)
            end_idx = nw_idx[-1] + (w_edge_mins * 2)
            if start_idx < 0: start_idx = 0
            if end_idx >= len(y): end_idx = len(y) - 1
            select_idx = np.arange(start_idx, end_idx + 1)
            print("Data before selection: {}, {}".format(x.shape, y.shape))
            x = x[select_idx]
            y = y[select_idx]
            print("Data after selection: {}, {}".format(x.shape, y.shape))

            # Saving as numpy files
            filename = os.path.basename(edf_fnames[file_id]).replace(".edf",  ".npz")
            save_dict = {
                "x": x,
                "y": y,
                "fs": sampling_rate
            }
            np.savez(os.path.join(args.output_dir, filename), **save_dict)
            print(" ---------- Done this file ---------")
            
        except Exception as e:
            print(f"Error processing file {edf_fnames[file_id]}: {e}")

if __name__ == "__main__":
    main()