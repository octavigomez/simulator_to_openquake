"""

This code extracts ruptures from RSQSim catalogues above a Mw threshold
and comptes a KS tests to analyze Poissonian behavior of the catalogue.

IMPORTANT: Please, put the path of your catalogue in the "path_in" variable.
Make sure that the catalogues you want to analyze are stored in
separated folders within the path_in route. Make sure also that within the
catalogue folders you have stored the fault model (in csv, delimited by ';'
and coordinates in km).

Authors: Octavi Gómez-Novell
Institution: CN Instituto Geológico y Minero de España (IGME-CSIC)

Location: Madrid, Spain
Last update: October 2025

License: CC-BY-NC 4.0

"""

# ============================================================
# IMPORTS AND GLOBAL SETTINGS
# ============================================================
import os
import json
import natsort
import warnings
import numpy as np
import pandas as pd
import utm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import kstest
from lib.parser_XML import parse_ruptures, create_xml
from pyproj import Transformer
# ------------------------------------------------------------
# WARNINGS AND PATH SETTINGS
# ------------------------------------------------------------
warnings.filterwarnings("ignore")
path = os.getcwd()
path = os.path.abspath(os.path.join(path, "../../"))
path_in = "C:/Users\Octavi Gomez\OneDrive - INSTITUTO GEOLÓGICO Y MINERO DE ESPAÑA\Software\simulator_to_openquake/diss_faults/khatereh"

# ============================================================
# GLOBAL PARAMETERS
# ============================================================
it = -1
m_filtering = 5.5    # Magnitude threshold for rupture extraction
cut_year = 5000
time_windows = [100000]
patch_threshold = 0

# ============================================================
# MAIN LOOP OVER FOLDERS
# ============================================================
for folder_name in natsort.natsorted(os.listdir(path_in)):
    folder = os.path.join(path_in, folder_name)
    print(folder)
    if not os.path.isdir(folder):
        continue
    print(folder)

    # ------------------------------------------------------------
    # READ INPUT CSV GEOMETRY
    # ------------------------------------------------------------
    file = os.path.join(folder, "Fault_model.txt")
    input_file = pd.read_csv(file, delimiter="\t", header=None)
    n = 1

    x1 = input_file.iloc[:, 0] * n
    x2 = input_file.iloc[:, 3] * n
    x3 = input_file.iloc[:, 6] * n
    y1 = input_file.iloc[:, 1] * n
    y2 = input_file.iloc[:, 4] * n
    y3 = input_file.iloc[:, 7] * n
    z1 = input_file.iloc[:, 2] * n
    z2 = input_file.iloc[:, 5] * n
    z3 = input_file.iloc[:, 8] * n

    file = np.column_stack((x1, x2, x3, y1, y2, y3, z1, z2, z3, np.array(range(1, len(input_file) + 1))))

    x_center = (x1 + x2 + x3) / 3
    y_center = (y1 + y2 + y3) / 3
    z_center = (z1 + z2 + z3) / 3

    transformer = Transformer.from_crs("EPSG:32633","EPSG:4326",  always_xy=True)
    x_center_deg, y_center_deg = transformer.transform(x_center, y_center)

    x_center_deg = np.round(x_center_deg, 4)
    y_center_deg = np.round(y_center_deg, 4)
    z_depth = z_center / -1000
    z_center_deg = np.round(z_depth, 2)

    # ============================================================
    # LOOP OVER TIME WINDOWS
    # ============================================================
    for tv in time_windows:
        time_window = tv
        mpl.rcParams['agg.path.chunksize'] = 100000

        # ------------------------------------------------------------
        # READ EVENT CATALOG
        # ------------------------------------------------------------

        file = os.path.join(folder, "Catalog.txt")
        file_raw = os.path.join(folder, "Cat_patch_info.txt")
        it = it + 1
        with open(file, "r") as file:
            catalog = pd.read_csv(file, delimiter="\t", header=None)
            catalog_raw = pd.read_csv(file_raw, delimiter="\t", header=None)

        # Extract parameters from catalogue and define magnitude range. Exlude first 2000 events of catalog
        if len(list(catalog.iloc[:, 1])) > 0:
            M0_ini = list(catalog.iloc[:, 1])[0:]
            M_ini = list(catalog.iloc[:, 0])[0:]
            x_ini = list(catalog.iloc[:, 4])[0:]
            y_ini = list(catalog.iloc[:, 5])[0:]
            z_ini = list(catalog.iloc[:, 6])[0:]
            area_ini = list(catalog.iloc[:, 2])[0:]
            t0_ini = list(catalog.iloc[:, 3])[0:]

        loc_cut = np.where(np.array(t0_ini) > cut_year)[0][0]
        M_ini = np.array(M_ini)[loc_cut:]
        M_ini = [float(i) for i in M_ini]
        M0_ini = np.array(M0_ini)[loc_cut:]
        M0_ini = [float(i) for i in M0_ini]
        area_ini = np.array(area_ini)[loc_cut:]
        num_events_ini = list(range(1, len(M_ini) + 1)) + loc_cut
        x_ini = np.array(x_ini)[loc_cut:]
        y_ini = np.array(y_ini)[loc_cut:]
        z_ini = np.array(z_ini)[loc_cut:]
        t0_ini = np.array(t0_ini)[loc_cut:]

        eList_ini = np.array(catalog_raw.iloc[:, 0])
        pList_ini = np.array(catalog_raw.iloc[:, 1])
        dList_ini = np.array(catalog_raw.iloc[:, 2])


        # ------------------------------------------------------------
        # CUT CATALOGUE AND FILTER SMALL EVENTS
        # ------------------------------------------------------------
        loc_list_cut = np.where(eList_ini == num_events_ini[0])[0][0]
        eList_ini = eList_ini[loc_list_cut:]
        pList_ini = pList_ini[loc_list_cut:]
        dList_ini = dList_ini[loc_list_cut:]

        ev, _, counts_events= np.unique(eList_ini, return_index=True, return_counts=True)
        locs_filter = np.where(counts_events > patch_threshold)[0]
        ev_final = ev[locs_filter]
        #ev_final = eList_ini[ev_final]
        ev_as_idx = locs_filter
        eList_inid = eList_ini
        eList_ini = eList_ini[np.isin(eList_ini, ev_final)]
        locs_all = np.isin(eList_inid, ev_final)
        pList_ini = pList_ini[locs_all]
        dList_ini = dList_ini[locs_all]

        M_ini = np.array(M_ini)[ev_as_idx]
        M0_ini = np.array(M0_ini)[ev_as_idx]
        area_ini = np.array(area_ini)[ev_as_idx]
        t0_ini = np.array(t0_ini)[ev_as_idx]
        x_ini = np.array(x_ini)[ev_as_idx]
        y_ini = np.array(y_ini)[ev_as_idx]
        z_ini = np.array(z_ini)[ev_as_idx]
        num_events_ini = np.array(num_events_ini)[ev_as_idx]

        # ------------------------------------------------------------
        # COMPLETENESS ANALYSIS
        # ------------------------------------------------------------
        mag_range = np.arange(0, 9, 0.1)
        hist, bins = np.histogram(M_ini, bins=mag_range)
        Mc = bins[np.argmax(hist)]
        idx_Mc = [x > Mc for x in M_ini]

        M_ini = np.array(M_ini)[idx_Mc]
        M0_ini = np.array(M0_ini)[idx_Mc]
        area_ini = np.array(area_ini)[idx_Mc]
        num_events_ini = np.array(num_events_ini)[idx_Mc]
        t0_ini = np.array(t0_ini)[idx_Mc]

        idx_Mc_all = np.where(np.isin(eList_ini, num_events_ini))[0]
        eList_ini = np.float64(eList_ini[idx_Mc_all])
        pList_ini = np.float64(pList_ini[idx_Mc_all])
        dList_ini = np.float64(dList_ini[idx_Mc_all])

        # ============================================================
        # TIME WINDOW SELECTION
        # ============================================================
        time_start = min(t0_ini)
        time_end = time_start + time_window
        time_loc = np.where((t0_ini >= time_start) & (t0_ini <= time_end))[0]
        t0 = np.array(t0_ini)[time_loc]
        print(f"Extracting ruptures from {min(t0)} to {max(t0)} yr")

        # ------------------------------------------------------------
        # FILTER EVENTS WITHIN WINDOW
        # ------------------------------------------------------------
        M = np.array(M_ini)[time_loc]
        M0 = np.array(M0_ini)[time_loc]
        area = np.array(area_ini)[time_loc]
        num_events = np.array(num_events_ini)[time_loc]

        hist, bins = np.histogram(M, bins=mag_range)
        Mc = bins[np.argmax(hist)]
        idx_Mc = [x > Mc for x in M]
        M = M[idx_Mc]
        M0 = M0[idx_Mc]
        area = area[idx_Mc]
        num_events = num_events[idx_Mc]
        t0 = t0[idx_Mc]

        idx_Mc_all = np.where(np.isin(eList_ini, num_events))[0]
        eList = np.float64(eList_ini[idx_Mc_all])
        pList = np.float64(pList_ini[idx_Mc_all])
        dList = np.float64(dList_ini[idx_Mc_all])

        # ============================================================
        # RUPTURE EXTRACTION
        # ============================================================
        M_filter_id = np.where(M >= m_filtering)[0]
        M_filter = M[M_filter_id]
        num_events_filt = num_events[M_filter_id]
        timing = t0[M_filter_id]

        table_rupture = []
        for c, i in enumerate(M_filter):
            eList_filt = np.where(eList == num_events_filt[c])[0]
            if len(eList_filt) >= 3:
                pList_fil = pList[eList_filt] - 1
                pList_fil = [int(y) for y in pList_fil]
                rate = 1 / time_window
                x_patch = x_center_deg[pList_fil]
                y_patch = y_center_deg[pList_fil]
                z_patch = np.array(z_center_deg[pList_fil])
                all_xyz = np.array([[x_patch[j],y_patch[j], z_patch[j]] for j in range(len(x_patch))]).flatten()
                rake = np.mean(input_file.iloc[pList_fil, 9])
                table_rupture.append([c + 1, i, rate, rake, " ".join(map(str, all_xyz))])

        table_rupture = np.array(table_rupture, dtype=object)


        # ============================================================
        # POISSON TEST
        # ============================================================

        cts = timing
        x = np.linspace(timing[0], timing[-1], len(timing))
        D, p_v = kstest(cts, np.linspace(timing[0], timing[-1], len(timing)))
        print("KStest test p-value:", p_v)

        # ------------------------------------------------------------
        # PLOT RESULTS
        # ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(15 / 2.5, 9 / 2.5), dpi=600)
        rd_seed = 100
        p_vs = np.zeros(rd_seed)
        D_s = np.zeros(rd_seed)
        random_timings = np.zeros((rd_seed, len(timing)))
        y_val = np.arange(1, len(x) + 1) / len(x)
        for i in range(rd_seed):
            random_timings[i, :] = min(timing) + (max(timing) - min(timing)) * np.random.rand(len(x))
            D_s[i], p_vs[i] = kstest(random_timings[i], np.linspace(timing[0], timing[-1], len(timing)))
            ax.plot(sorted(random_timings[i, :]), y_val, c="grey", alpha=0.8,
                    label="Random poissonian catalogues" if i == 0 else None)

        percentile = np.percentile(p_vs, [2.5, 97.5])
        ax.plot(timing, y_val, color="red", label="Simulated catalogue")
        ax.plot(x, y_val, color="black", label="Poissonian catalogue")
        ax.text(45000, .2, f"p-value = {round(p_v, 2)}\n 2.5 - 97.5% percentile range: "
                f"{round(percentile[0], 2)}-{round(percentile[1], 2)}", ha="center")
        ax.set_xlabel("Year")
        ax.set_ylabel("Probability")
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid("on")
        fig.show()

        # ------------------------------------------------------------
        # FINAL ASSESSMENT
        # ------------------------------------------------------------
        if percentile[0] < p_v < percentile[1]:
            print("Catalogue is poissonian")
        else:
            print("Catalogue is not poissonian")

        # ------------------------------------------------------------
        # PARSE THE RUPTURES AND CONVERT THEM TO XML FOR OPENQUAKE
        # -----------------------------------------------------------
        # The xml file with the ruptures will be stored in the path defined
        # in the "path_in" variable.

        ruptures = parse_ruptures(table_rupture)
        create_xml(ruptures, folder + '/ruptures'+str(it)+'.xml')

        print(f"Successfully converted {len(ruptures)} ruptures to XML format.")