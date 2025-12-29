import mne
import numpy as np
import pandas as pd
import pycrostates as pcs
from pathlib import Path
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
)

def group_gfp_extr_BIDS(
    bids_paths:list[BIDSPath],
    min_peak_distance:int = 1,
    n_samples:int = 100,
) -> pcs.io.ChData:
    """"Extract GFP peaks from multiple BIDS paths and equalize the number of peaks across subjects."""
    individual_gfp_peaks = list()
    for bids_path in bids_paths:
        # load Data
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.load_data()
        raw.resample(250, npad="auto")
        raw.set_eeg_reference("average")
        # extract GFP peaks
        gfp_peaks = pcs.preprocessing.extract_gfp_peaks(raw, min_peak_distance=min_peak_distance)
        # equalize peak number across subjects by resampling
        if n_samples:
            gfp_peaks_r = pcs.preprocessing.resample(gfp_peaks, n_resamples=10, n_samples=n_samples, random_state=42)
            subject_peaks = np.hstack([r.get_data() for r in gfp_peaks_r])
        else:
            subject_peaks = gfp_peaks.get_data()
        individual_gfp_peaks.append(subject_peaks)

    individual_gfp_peaks = np.hstack(individual_gfp_peaks)
    individual_gfp_peaks = pcs.io.ChData(individual_gfp_peaks, raw.info)

    return individual_gfp_peaks

if __name__ == "__main__":
    patient_info_path = Path(r"H:\msy\dalunwen\info_00_patient.xlsx")
    bids_root = Path(r"H:\msy\dalunwen\data\data_00_BIDS_eeg")

    patient_info = pd.read_excel(patient_info_path)
    sessions = get_entity_vals(bids_root, "session")

    subject = [i[3:5] for i in patient_info[patient_info['诊断'] == '缺血缺氧脑病']['ID']]
    session = sessions[0]
    datatype = "eeg"
    task = "RestEyesClosed"
    extensions = [".edf"]  

    bids_paths = find_matching_paths(
        bids_root, subjects=subject, sessions=session, datatypes=datatype, tasks=task ,extensions=extensions
    )
    group_gfp_peaks = group_gfp_extr_BIDS(bids_paths, n_samples=100)
    adjacency, ch_names = mne.channels.find_ch_adjacency(info=group_gfp_peaks.info, ch_type="eeg")
    #pcs.preprocessing.apply_spatial_filter(gfp, n_jobs=-1)

    n_clusters = 4
    ModK = pcs.cluster.ModKMeans(n_clusters=n_clusters, random_state=42)
    ModK.fit(group_gfp_peaks, n_jobs=5)
    ModK.plot()
    ModK.reorder_clusters(order=[0,2,1,3])
    ModK.rename_clusters(new_names=["A", "B", "C", "D"])
    ModK.plot()
    ModK.invert_polarity([False, False, True, True])
    ModK.plot()
    ModK.save(r"H:\msy\dalunwen\data\data_22_microstate\microstate_qxqy_4clust.fif")