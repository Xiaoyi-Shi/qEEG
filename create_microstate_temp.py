import mne
import numpy as np
import pycrostates as ps
from pathlib import Path
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
)

bids_root = Path(r"H:\msy\dalunwen\data\data_00_BIDS_eeg")
sessions = get_entity_vals(bids_root, "session", ignore_sessions="on")
datatype = "eeg"
extensions = [".edf", ".tsv"]  # ignore .json files
bids_paths = find_matching_paths(
    bids_root, datatypes=datatype, sessions=sessions, extensions=extensions
)

task = "RestEyesClosed"
suffix = "eeg"
subject = "05"
bids_path = BIDSPath(root=bids_root, session=sessions[0], datatype=datatype)
bids_path = bids_path.update(subject=subject, task=task, suffix=suffix)

raw = read_raw_bids(bids_path=bids_path, verbose=False)
raw.load_data()
raw.plot()

raw.resample(250, npad="auto")
raw.set_eeg_reference("average")

adjacency, ch_names = mne.channels.find_ch_adjacency(info=raw.info, ch_type="eeg")
gfp = ps.preprocessing.extract_gfp_peaks(raw)
#ps.preprocessing.apply_spatial_filter(gfp, n_jobs=-1)

n_clusters = 4
ModK = ps.cluster.ModKMeans(n_clusters=n_clusters, random_state=42)
ModK.fit(gfp, n_jobs=5)
ModK.plot()
ModK.reorder_clusters(order=[0,3,2,1])
ModK.rename_clusters(new_names=["A", "B", "C", "D"])
ModK.plot()
ModK.invert_polarity([False, False, True, False])
ModK.plot()
ModK.save(r"H:\msy\dalunwen\data\data_22_qEEG\microstate_model_4clust.fif")

segmentation = ModK.predict(
    raw,
    reject_by_annotation=True,
    factor=10,
    half_window_size=10,
    min_segment_length=5,
    reject_edges=True,
)

parameters = segmentation.compute_parameters()
parameters

x = ModK.cluster_names
y = [parameters[elt + "_gev"] for elt in x]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel("Microstates")
ax.set_ylabel("Global explained Variance (ratio)")
plt.show()