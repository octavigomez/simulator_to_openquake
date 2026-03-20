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
Last update: January 2026

License: CC-BY-NC 4.0

"""

# ============================================================
# IMPORTS AND GLOBAL SETTINGS
# ============================================================
import os
import natsort
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from lib.parser_XML import parse_ruptures, create_xml
from pyproj import Transformer
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
from scipy import stats

# ------------------------------------------------------------
# WARNINGS AND PATH SETTINGS
# ------------------------------------------------------------
warnings.filterwarnings("ignore")
path = os.getcwd()
path = os.path.abspath(os.path.join(path, "../../"))
path_in = "C:/Users\Octavi Gomez\OneDrive - INSTITUTO GEOLÓGICO Y MINERO DE ESPAÑA\Software/MCQsim_to_ASCII_update\Simulation_models/diss/diss_rough"

# ============================================================
# GLOBAL PARAMETERS
# ============================================================
it = -1
m_filtering = 6    # Minimum magnitude threshold for rupture extraction
cut_year = 50000   # Year of start of the catalogue
time_windows = [50000] # Length of the catalogue (years) you want to extract
patch_threshold = 0 # Default
percentage_filter = 0 # Percentage of slip to be filtered out
distance_clust = 3000 # Minimum distance to search for points belonging to a cluster
num_samples = 2 # Minimum number of patches to define a cluster
fA = 0.0


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


    def triangle_area_3d(p1, p2, p3):
        """
        Calculate the area of a triangle in 3D space.

        Parameters:
        p1, p2, p3: array-like, shape (3,)
            The (x, y, z) coordinates of the three vertices

        Returns:
        float: The area of the triangle
        """
        # Convert to numpy arrays
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)

        # Calculate two edge vectors
        v1 = p2 - p1
        v2 = p3 - p1

        # Calculate cross product
        cross = np.cross(v1, v2)

        # Area is half the magnitude of the cross product
        area = 0.5 * np.linalg.norm(cross)

        return area


    # Example usage
    areas_tris = []
    for i in range(len(x1)):
        point1 = [x1[i], y1[i], z1[i]]
        point2 = [x2[i], y2[i], z2[i]]
        point3 = [x3[i], y3[i], z3[i]]

        area = triangle_area_3d(point1, point2, point3)
        areas_tris.append(area)
    areas_tris = np.array(areas_tris)

    file = np.column_stack((x1, x2, x3, y1, y2, y3, z1, z2, z3, np.array(range(1, len(input_file) + 1))))

    x_center = (x1 + x2 + x3) / 3
    y_center = (y1 + y2 + y3) / 3
    z_center = (z1 + z2 + z3) / 3

    transformer = Transformer.from_crs("EPSG:3035","EPSG:4326",  always_xy=True)
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
        tList_ini = np.array(catalog_raw.iloc[:, 3])


        # ------------------------------------------------------------
        # CUT CATALOGUE AND FILTER SMALL EVENTS
        # ------------------------------------------------------------
        loc_list_cut = np.where(eList_ini == num_events_ini[0])[0][0]
        eList_ini = eList_ini[loc_list_cut:]
        pList_ini = pList_ini[loc_list_cut:]
        dList_ini = dList_ini[loc_list_cut:]
        tList_ini = tList_ini[loc_list_cut:]

        ev, _, counts_events = np.unique(eList_ini, return_index=True, return_counts=True)
        locs_filter = np.where(counts_events > patch_threshold)[0]
        ev_final = ev[locs_filter]
        #ev_final = eList_ini[ev_final]
        ev_as_idx = locs_filter
        eList_inid = eList_ini
        eList_ini = eList_ini[np.isin(eList_ini, ev_final)]
        locs_all = np.isin(eList_inid, ev_final)
        pList_ini = pList_ini[locs_all]
        dList_ini = dList_ini[locs_all]
        tList_ini = tList_ini[locs_all]

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
        tList_ini = np.float64(tList_ini[idx_Mc_all])

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
        tList = np.float64(tList_ini[idx_Mc_all])

        # ============================================================
        # RUPTURE EXTRACTION AND FILTERING BY SLIP THRESHOLD
        # ============================================================
        M_filter_id = np.where(M >= m_filtering)[0]
        M_filter = M[M_filter_id]
        num_events_filt = num_events[M_filter_id]
        timing = t0[M_filter_id]
        m0_filter = M0[M_filter_id]

        table_rupture = []
        changes = []
        magnitudes = []
        magnitudes_filt =[]
        all_removed = []

        all_moment_percs = []

        ev = 0
        for c, i in enumerate(M_filter):
            eList_filt = np.where(eList == num_events_filt[c])[0]
            #print(i)
            if len(eList_filt) >= 3:
                dList_fil = dList[eList_filt]
                sl_thr = np.mean(dList_fil)*percentage_filter
                loc_slip = np.where(dList_fil >= sl_thr)[0]
                loc_slip_min = np.where(dList_fil<sl_thr)[0]
                eList_filt2 = eList_filt[loc_slip]
                pList_fil = pList[eList_filt2] - 1
                pList_fil = [int(y) for y in pList_fil]
                area_rupture = np.sum(areas_tris[pList_fil])
                tList_filt = tList[eList_filt2]
                rate = 1 / time_window
                x_patch = x_center_deg[pList_fil]
                y_patch = y_center_deg[pList_fil]
                z_patch = np.array(z_center_deg[pList_fil])
                x_patch_utm = x_center[pList_fil]
                y_patch_utm = y_center[pList_fil]
                eList_filt3 = eList_filt[loc_slip_min]
                pList_fil_min = pList[eList_filt3] - 1
                pList_fil_min = [int(y) for y in pList_fil_min]
                tList_filt_min = tList[eList_filt3]
                x_patch_min = x_center_deg[pList_fil_min]
                y_patch_min = y_center_deg[pList_fil_min]


                # ============================================================
                # PERFORM CLUSTER ANALYSIS FOR FURTHER CLEANING
                # ============================================================

                X = np.column_stack((x_patch_utm, y_patch_utm))
                dbscan = DBSCAN(eps=distance_clust, min_samples=num_samples)
                labels = dbscan.fit_predict(X)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                # Find cluster with highest area




                #Area and moment analysis

                labs = [-1]
                area_clusters = []
                moment_clusters = []
                for lab in np.unique(labels[labels!=-1]):
                    loc_lab = np.where(labels == lab)[0]
                    moment_late = 3e10*np.sum(areas_tris[pList_fil][loc_lab]*dList_fil[loc_slip][loc_lab])
                    moment_ini = 3e10*np.sum(areas_tris[pList_fil]*dList_fil[loc_slip])
                    area_clusters.append(np.sum(areas_tris[pList_fil][loc_lab]))

                    perc_area = np.sum(areas_tris[pList_fil][loc_lab])/np.sum(areas_tris[pList_fil])
                    perc_area = moment_late/moment_ini
                    moment_clusters.append(moment_late)
                    if perc_area<fA:
                        labs.append(lab)
                        #print("Removed cluster "+str(lab)+" with perc. area "+str(perc_moment*100) + "%")
                # perc_area_max = area_clusters/np.sum(area_clusters)
                # perc_moment_max = moment_clusters/max(moment_clusters)
                # loc_perc = np.where(perc_area_max<fA)[0]
                # labs.append(labels[labels!=-1][loc_perc])
                # labs = np.concatenate(([labs[0]], labs[1]))

                cluster_loc = np.where(~np.isin(labels, labs))[0]
                noise_loc = np.where(np.isin(labels, labs))[0]

                # Recompute magnitude without noise

                avg_slip_filt = dList_fil[loc_slip][cluster_loc]
                moment_filt = np.sum(3e10 * avg_slip_filt * areas_tris[pList_fil][cluster_loc])
                magnitude_filt = 2 / 3 * (np.log10(moment_filt) - 9.1)
                magnitudes.append(M_filter[c])
                magnitudes_filt.append(magnitude_filt)
                magnitude_change = M_filter[c] - magnitude_filt
                changes.append(magnitude_change)
                patches_removed = len(pList_fil_min) + len(noise_loc)
                all_removed.append(patches_removed)
                perc_moment = moment_filt / m0_filter[c]
                all_moment_percs.append(perc_moment)





                if magnitude_filt >= m_filtering:
                    x_patch1 = x_center_deg[pList_fil][cluster_loc]
                    y_patch1 = y_center_deg[pList_fil][cluster_loc]
                    z_patch1 = np.array(z_center_deg[pList_fil])[cluster_loc]
                    all_xyz1 = np.array(
                        [[x_patch1[j], y_patch1[j], z_patch1[j]] for j in range(len(x_patch1))]).flatten()
                    rake1 = np.mean(np.array(input_file.iloc[pList_fil, 9])[cluster_loc])
                    table_rupture.append([c + 1, magnitude_filt, rate, rake1, " ".join(map(str, all_xyz1))])

                #print("Patches removed:", len(dList_fil)-len(cluster_loc))

                # ===> Activate this in case you want to visualize ruptures and filtering applied

                #

                fig, ax = plt.subplots()
                plt.scatter(x_center_deg, y_center_deg, c="grey")
                cbar = plt.scatter(x_patch[noise_loc], y_patch[[noise_loc]], s=2, c= "red")
                # plt.scatter(x_patch_min, y_patch_min, c="red", s=2, label = "removed")
                plt.scatter(x_center_deg[pList_fil][cluster_loc], y_center_deg[pList_fil][cluster_loc], c=dList_fil[loc_slip][cluster_loc],
                            label="removed-fA", s=2)
                plt.title("Min slip = " + str(sl_thr)+"m")
                plt.colorbar(cbar)
                plt.legend()
                os.makedirs(folder+"/ruptures", exist_ok=True)
                fig.show()
                fig.savefig(folder + '/ruptures/ruptures_unfilt'+str(c)+'.png')

                #
                # # ==> Activate if you want to visualize the results of cluster analysis
                #
                # plt.figure(figsize=(10, 6))
                # unique_labels = set(labels)
                # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
                # for label, color in zip(unique_labels, colors):
                #     if label == -1:
                #         # Black color for noise points
                #         color = [0, 0, 0, 1]
                #
                #     class_member_mask = (labels == label)
                #     xy = X[class_member_mask]
                #     plt.scatter(xy[:, 0], xy[:, 1], c=[color], s=20,
                #                 label=f'Cluster {label}' if label != -1 else 'Noise', edgecolors=None)
                #
                #
                # plt.title('DBSCAN Clustering'+ "rupture:"+str(c+1))
                # plt.legend()
                # plt.show()


        table_rupture = np.array(table_rupture, dtype=object)


        # ------------------------------------------------------------
        # PARSE THE RUPTURES AND CONVERT THEM TO XML FOR OPENQUAKE
        # -----------------------------------------------------------
        # The xml file with the ruptures will be stored in the path defined
        # in the "path_in" variable.

        ruptures = parse_ruptures(table_rupture)
        create_xml(ruptures, folder + '/ruptures_unfilt.xml')

        print(f"Successfully converted {len(ruptures)} ruptures to XML format.")


        #FINAL MFD CHECK

        hist_ini, _ = np.histogram(M, bins=mag_range)
        hist_ini_modif,_ = np.histogram(magnitudes_filt, bins=mag_range)

        rate_ini = hist_ini/time_window
        rate_after = hist_ini_modif/time_window

        rate_ini_thr = rate_ini[mag_range[:-1]>=m_filtering]
        rate_after_thr = rate_after[mag_range[:-1] >= m_filtering]

        rate_loss = (1 - (np.sum(rate_after_thr)/np.sum(rate_ini_thr)))*100
        rate_perc_change = ((np.sum(rate_after_thr)-np.sum(rate_ini_thr))/np.sum(rate_ini_thr))*100
        print("Total EQ rate loss = ", rate_loss)
        print("Total EQ rate % change = ", rate_perc_change)



        fig2, ax2 = plt.subplots()
        plt.plot(mag_range[:-1], np.flip(np.cumsum(rate_ini[::-1])), label='initial-cum')
        plt.plot(mag_range[:-1], np.flip(np.cumsum(rate_after[::-1])), label='filtered-cum')
        plt.plot(mag_range[:-1], rate_ini, label='initial', linestyle='dashed')
        plt.plot(mag_range[:-1], rate_after, label='filtered', linestyle='dashed')
        plt.legend()
        plt.yscale('log')
        plt.xlim(min(M), 8)
        fig2.show()
