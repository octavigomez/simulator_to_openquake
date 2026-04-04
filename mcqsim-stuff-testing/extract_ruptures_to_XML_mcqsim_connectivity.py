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
num_samples = 1 # Minimum number of patches to define a cluster

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
    file = os.path.join(folder, "Fault_model_modif.txt")
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
        x_ini = x_ini[idx_Mc]
        y_ini = y_ini[idx_Mc]
        z_ini = z_ini[idx_Mc]

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
        x = x_ini[time_loc]
        y = y_ini[time_loc]
        z = z_ini[time_loc]

        hist, bins = np.histogram(M, bins=mag_range)
        Mc = bins[np.argmax(hist)]
        idx_Mc = [x > Mc for x in M]
        M = M[idx_Mc]
        M0 = M0[idx_Mc]
        area = area[idx_Mc]
        num_events = num_events[idx_Mc]
        t0 = t0[idx_Mc]
        x = x_ini[idx_Mc]
        y = y_ini[idx_Mc]
        z = z_ini[idx_Mc]

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
        table_rupture1 = []
        changes = []
        magnitudes = []
        magnitudes_filt =[]
        all_removed = []
        all_slips_removed = []

        all_magnitudes= []
        all_areas= []
        all_times= []
        all_moms= []
        all_eList_re= []
        all_pList_re= []
        all_dList_re= []
        all_tList_re = []
        all_moment_percs = []
        all_x = []
        all_y = []
        all_z = []
        centroid_x_re_all =[]
        centroid_y_re_all =[]
        centroid_z_re_all =[]

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
                all_slips_removed.append(dList[eList_filt3])


                # ============================================================
                # PERFORM CLUSTER ANALYSIS FOR FURTHER CLEANING
                # ============================================================

                X = np.column_stack((x_patch_utm, y_patch_utm))
                dbscan = DBSCAN(eps=distance_clust, min_samples=num_samples)
                labels = dbscan.fit_predict(X)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                cluster_loc = np.where(labels != -1)[0]
                noise_loc = np.where(labels == -1)[0]

                # Recompute just for noise

                avg_slip_filt = dList_fil[loc_slip][cluster_loc]
                moment_filt = np.sum(3e10*avg_slip_filt * areas_tris[pList_fil][cluster_loc])
                magnitude_filt = 2 / 3 * (np.log10(moment_filt) - 9.1)
                magnitudes.append(M_filter[c])
                magnitudes_filt.append(magnitude_filt)
                magnitude_change = M_filter[c] - magnitude_filt
                changes.append(magnitude_change)
                patches_removed = len(pList_fil_min) + len(noise_loc)
                all_removed.append(patches_removed)

                perc_moment = moment_filt/m0_filter[c]
                all_moment_percs.append(perc_moment)
                #print("Moment_percentage:", np.mean(all_moment_percs))

                #print(max(changes))
                #print("Patches removed:", len(dList_fil)-len(cluster_loc))

                # ===> Activate this in case you want to visualize ruptures and filtering applied

                #
                # #plt.scatter(x_center_deg, y_center_deg, c="grey")
                # cbar = plt.scatter(x_patch, y_patch, c=dList_fil[loc_slip], s=2)
                # # plt.scatter(x_patch_min, y_patch_min, c="red", s=2, label = "removed")
                # plt.scatter(x_center_deg[pList_fil][noise_loc], y_center_deg[pList_fil][noise_loc], c="red",
                #             label="removed-clust", s=2)
                # plt.title("Min slip = " + str(sl_thr)+"m")
                # plt.colorbar(cbar)
                # plt.legend()
                # plt.show()

                # fig, ax = plt.subplots()
                # cbar = plt.scatter(x_patch[noise_loc], y_patch[[noise_loc]], s=2, c="red")
                # # plt.scatter(x_patch_min, y_patch_min, c="red", s=2, label = "removed")
                # plt.scatter(x_center_deg[pList_fil][cluster_loc], y_center_deg[pList_fil][cluster_loc],
                #             c=dList_fil[loc_slip][cluster_loc],
                #             label="removed-fA", s=2)
                # plt.title("Min slip = " + str(sl_thr) + "m")
                # plt.colorbar(cbar)
                # plt.legend()
                # os.makedirs(folder + "/ruptures", exist_ok=True)
                # fig.savefig(folder + '/ruptures/ruptures_unfilt' + str(c) + '.png')
                #
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


                # Inter and intracluster distance analysis

                XYZ = np.column_stack((x_patch_utm, y_patch_utm, z_patch))
                filter_label = np.unique(labels[labels!=-1])
                clusters = []

                if len(filter_label) > 0:
                    intra_distance = []
                    for r in filter_label:
                        XYZ_filt = XYZ[labels==r]
                        clusters.append(XYZ_filt)
                        distance = pdist(XYZ_filt, metric='euclidean')
                        intra_distance.append(list(distance))
                        #plt.hist(distance, bins=10, density=True)
                        # kde = stats.gaussian_kde(distance)
                        # rg = np.linspace(0, 1000000, 5000)
                        # scaled = kde(rg)
                    #     plt.plot(rg[scaled>1e-10], scaled[scaled>1e-10])
                    # plt.title("Intra-cluster distance")
                    # plt.show()

                    intra_distance_flat = np.array([x for row in intra_distance for x in row])
                    threshold = np.percentile(intra_distance_flat, 100)
                    #print(threshold)


                    #ARA TOCA ANALITZAR CONNECTIVITAT PER ANALITZAR SI REALMENT ES MULTI-FAULT O NO
                    # Si en 3 clusters dos d'ells tenen una distancia curta i una gran, si les curtes tenen en comu el mateix cluster
                    # es considera connectivitat:


                    store_dist = []
                    store_combos = []
                    store_dist_distr = []
                    all_distances_inter = []
                    if len(clusters)>1:
                        for cl in range(len(clusters)-1):
                            cluster_ini = clusters[cl]
                            for cl1 in range(1, len(clusters)):
                                if cl!=cl1 and cl<cl1:
                                    store_combos.append([cl, cl1])
                                    cluster_last = clusters[cl1]
                                    distances_inter = []
                                    for dist in range(len(cluster_ini)):
                                        for dist2 in range(len(cluster_last)):
                                            pt1 = cluster_ini[dist]
                                            pt2 = cluster_last[dist2]
                                            dist_clust = np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2+(pt1[2]-pt2[2])**2)
                                            distances_inter.append(dist_clust)
                                    store_dist.append(np.min(distances_inter))
                                    store_dist_distr.append(distances_inter)
                                    # kde_clust = stats.gaussian_kde(distances_inter)
                                    # scaled1 = kde_clust(rg)
                                    #plt.plot(rg[scaled1 > 1e-10], scaled1[scaled1 > 1e-10], label="cluster"+str(cl)+"-"+str(cl1))
                                    all_distances_inter.append(distances_inter)
                    else:
                        all_distances_inter.append(list(np.zeros(len(clusters[0]))))
                        store_dist.append([0])
                        store_dist_distr.append(list(np.zeros(len(clusters[0]))))
                        store_combos.append([0,0])
                    # plt.title("Inter-cluster distance")
                    # plt.legend()
                    # plt.show()


                    combo = np.column_stack((store_combos, store_dist))

                    #print(store_dist, threshold)

                    # plt.hist(intra_distance_flat, bins=50, color="grey", label="Intra-cluster distance")
                    # plt.plot([threshold, threshold], [0, max(np.histogram(intra_distance_flat, bins =50)[0])], c="red", label="median")
                    # for dist in range(len(store_dist_distr)):
                    #     plt.hist(store_dist_distr[dist], bins=50)
                    # plt.xlabel("Distance (m)")
                    # plt.title("Intracluster distance distribution vs inter-cluster / thr = "+ str(round(threshold)))
                    # plt.legend()
                    # plt.show()

                    # Connectivity analysis
                    G = nx.Graph()

                    nodes = np.unique(combo[:, :2])
                    G.add_nodes_from(nodes)
                    for i, j, d in combo:
                        if d <= threshold:
                            G.add_edge(int(i), int(j))
                    components = list(nx.connected_components(G))
                    components = [np.array([float (i) for i in j]) for j in components]
                    #print(components)
                    #
                    # plt.scatter(np.array(x_patch_utm)[cluster_loc], np.array(y_patch_utm)[cluster_loc], c="black")
                    # plt.show()


                    # ============================================================
                    # RECOMPUTE MAGNITUDES OF EXTRA EVENTS
                    # ============================================================

                    # Initialize new catalogue variables
                    mags = []
                    moms = []
                    areas = []
                    time = []
                    x_re = []
                    y_re = []
                    z_re = []
                    eList_re = []
                    pList_re = []
                    dList_re = []
                    tList_re = []
                    centroid_x_re = []
                    centroid_y_re = []
                    centroid_z_re = []

                    #plt.scatter(x_center_deg, y_center_deg, marker='o', s=2, c="grey", zorder=1)
                    for re in range(len(components)):
                        ev = ev+1
                        comp = components[re]
                        loc_cluster_n = np.isin(labels, comp)
                        area = areas_tris[pList_fil][loc_cluster_n]
                        area_sum = np.sum(area)
                        slip = dList_fil[loc_slip][loc_cluster_n]
                        mom = np.sum(area * slip) * 3e10
                        mag = round(2/3*(np.log10(mom)-9.1),2)
                        ts = np.min(tList_filt[loc_cluster_n])
                        loc_time = np.where(tList_filt[loc_cluster_n]==np.min(tList_filt[loc_cluster_n]))[0][0]
                        mags.append(mag)
                        moms.append(mom)
                        areas.append(area_sum)
                        time.append(ts)
                        sls = dList_fil[loc_slip][loc_cluster_n]
                        tms = tList_filt[loc_cluster_n]
                        pchs = np.array(pList_fil)[loc_cluster_n]
                        eList_re.append(np.zeros(len(sls))+ev)
                        pList_re.append(pchs)
                        dList_re.append(sls)
                        tList_re.append(tms)


                        # ============================================================
                        # PREPARE TABLE FOR RUPTURE EXPORT TO XML
                        # ============================================================

                        x_patch = x_center_deg[pList_fil][loc_cluster_n]
                        y_patch = y_center_deg[pList_fil][loc_cluster_n]
                        z_patch = np.array(z_center_deg[pList_fil][loc_cluster_n])

                        x_re.append(x_patch[loc_time])
                        y_re.append(y_patch[loc_time])
                        z_re.append(z_patch[loc_time])

                        centroid_x_re.append(np.mean(x_patch))
                        centroid_x_re.append(np.mean(y_patch))
                        centroid_z_re.append(np.mean(z_patch))


                        #plt.scatter(x_patch, y_patch, marker='o', s=10, zorder=2)


                        if mag>=m_filtering:
                            all_xyz = np.array(
                                [[x_patch[j], y_patch[j], z_patch[j]] for j in range(len(x_patch))]).flatten()
                            rake = np.mean(np.array(input_file.iloc[pchs-1, 9]))
                            table_rupture.append([ev, mag, rate, rake, " ".join(map(str, all_xyz))])


                    #plt.show()

                    if magnitude_filt>=m_filtering:
                        x_patch1 = x_center_deg[pList_fil][cluster_loc]
                        y_patch1 = y_center_deg[pList_fil][cluster_loc]
                        z_patch1 = np.array(z_center_deg[pList_fil])[cluster_loc]
                        all_xyz1 = np.array([[x_patch1[j],y_patch1[j], z_patch1[j]] for j in range(len(x_patch1))]).flatten()
                        rake1 = np.mean(np.array(input_file.iloc[pList_fil, 9])[cluster_loc])
                        table_rupture1.append([c+1, magnitude_filt, rate, rake1, " ".join(map(str, all_xyz1))])


                    all_magnitudes.append(mags)
                    all_areas.append(areas)
                    all_times.append(time)
                    all_moms.append(moms)
                    all_eList_re.append(eList_re)
                    all_pList_re.append(pList_re)
                    all_dList_re.append(dList_re)
                    all_tList_re.append(tList_re)
                    all_x.append(x_re)
                    all_y.append(y_re)
                    all_z.append(z_re)
                    centroid_x_re_all.append(centroid_x_re)
                    centroid_y_re_all.append(centroid_y_re)
                    centroid_z_re_all.append(centroid_z_re)



        all_slips_removed = [x for xs in all_slips_removed for x in xs]
        table_rupture = np.array(table_rupture, dtype=object)
        table_rupture1 = np.array(table_rupture1, dtype=object)

        #print(min([len(i) for i in all_pList_re]))

        all_magnitudes = np.array([x for row in all_magnitudes for x in row])
        all_areas = np.array([x for row in all_areas for x in row])
        all_times = np.array([x for row in all_times for x in row])
        all_moms = np.array([x for row in all_moms for x in row])
        all_x = np.array([x for row in all_x for x in row])
        all_y = np.array([x for row in all_y for x in row])
        all_z = np.array([x for row in all_z for x in row])
        centroid_x_re_all= np.array([x for row in centroid_x_re_all for x in row])
        centroid_y_re_all= np.array([x for row in centroid_y_re_all for x in row])
        centroid_z_re_all= np.array([x for row in centroid_z_re_all for x in row])
        all_eList_re = np.concatenate([x for row in all_eList_re for x in row])
        all_pList_re = np.concatenate([x for row in all_pList_re for x in row])
        all_dList_re = np.concatenate([x for row in all_dList_re for x in row])
        all_tList_re = np.concatenate([x for row in all_tList_re for x in row])


        all_final_magnitudes = np.concatenate((all_magnitudes, np.array(M)[np.array(M)<m_filtering]))
        hist_ini, _ = np.histogram(M, bins = mag_range)
        hist_after, _ = np.histogram(all_final_magnitudes, bins=mag_range)

        hist_ini_modif,_ = np.histogram(magnitudes_filt, bins=mag_range)
        print(hist_ini[mag_range[:-1]>=6])
        print(hist_ini_modif[mag_range[:-1]>=6])


        rate_ini = hist_ini/time_window
        rate_after = hist_after/time_window

        # b-value

        Mc_after = mag_range[np.argmax(hist_after)]
        Mc_after = m_filtering
        M_complete = all_final_magnitudes[all_final_magnitudes >= Mc_after]
        deltaM = 0.1
        b = (np.log10(np.exp(1))) / (np.mean(M_complete) - (Mc_after - deltaM/2))
        print("b-value = ", b)

        # Export both GR distributions

        GRs = np.column_stack((mag_range[:-1], rate_ini, rate_after))
        path_act = os.getcwd()
        os.makedirs(path_act+ "/outputs", exist_ok =True)
        np.savetxt(path_act+"/outputs"+"/filt_and_unfilt_GRs.txt", GRs)

        #Export previous and new catalogue

        old_cat = np.column_stack((magnitudes_filt, t0[M_filter_id]))
        new_cat = np.column_stack((all_magnitudes, all_times))
        np.savetxt(path_act+"/outputs"+"/old_Catalogue.txt", old_cat)
        np.savetxt(path_act + "/outputs" + "/new_Catalogue.txt", new_cat)



        plt.plot(mag_range[:-1], np.flip(np.cumsum(rate_ini[::-1])), label='initial-cum')
        plt.plot(mag_range[:-1], np.flip(np.cumsum(rate_after[::-1])), label='recalculated-cum')
        plt.plot(mag_range[:-1], rate_ini, label='initial', linestyle='dashed')
        plt.plot(mag_range[:-1], rate_after, label='recalculated', linestyle='dashed')
        plt.legend()
        plt.yscale('log')
        plt.xlim(min(M), 8)
        plt.title("b-value_filtered ="+str(round(b,2)))
        plt.show()



        #Comparison with scaling laws



        def Thingbaijam2017(self, kinematics):
            if kinematics == "N":
                a, b, sd_a, sd_b, sd_logL = -2.551, 0.808, 0.423, 0.059, 0.181
            elif kinematics == "R":
                a, b, sd_a, sd_b, sd_logL = -4.362, 1.049, 0.445, 0.066, 0.94
            elif kinematics == "SS":
                a, b, sd_a, sd_b, sd_logL = -3.486, 0.942, 0.399, 0.058, 0.184
            M = (np.log10(self/1e6) - a) / b
            sd = sd_logL / b
            return M, sd


        def Leonard2010(self, kinematics):
            if (kinematics == "N") | (kinematics == "R"):
                a, b, sd_b = 1.5, 6.10, [5.69, 6.6]
            elif kinematics == "SS":
                a, b, sd_b = 1.5, 6.09, [5.69, 6.47]
            elif kinematics == "SCR":
                a, b, sd_b = 1.5, 6.38, [6.22, 6.52]
            logM0 = a * np.log10(self) + b
            M = 2 / 3 * logM0 - 6.07
            sd_b_fixed = (sd_b[1] - sd_b[0]) / 2
            sd = (2 / 3) * sd_b_fixed
            return M, sd


        areas_range = np.linspace(min(all_areas), max(all_areas), 1000)
        M_Le_N, sd_Le_N = Leonard2010(areas_range, "N")
        M_Le_SS, sd_Le_SS = Leonard2010(areas_range, "SS")
        M_Le_SCR, sd_Le_SCR = Leonard2010(areas_range, "SCR")


        plt.plot(areas_range, M_Le_N, c="red",  label='Le10-dipSlip mean')
        #plt.plot(areas_range, M_Le_N + sd_Le_N, c="red", linestyle=":")
        #plt.plot(areas_range, M_Le_N - sd_Le_N, c="red", linestyle=":")
        plt.fill_between(areas_range,
                     M_Le_N - sd_Le_N*2,
                     M_Le_N + sd_Le_N*2,
                     alpha=0.1, color="red", label='Le10-dipSlip 2-sigma')

        plt.plot(areas_range, M_Le_SS, c="black",  label='Le10-strikeSlip mean')
        #plt.plot(areas_range, M_Le_SS + sd_Le_SS, c="black", linestyle=":")
        #plt.plot(areas_range, M_Le_SS - sd_Le_SS, c="black", linestyle=":")
        plt.fill_between(areas_range,
                     M_Le_SS - sd_Le_SS*2,
                     M_Le_SS + sd_Le_SS*2,
                     alpha=0.1, color="black",label='Le10-strikeSlip 2-sigma')

        plt.scatter(all_areas, all_magnitudes, s=6, c="black", alpha=0.5)
        plt.xscale("log")
        plt.legend()
        plt.ylim(6, 8)
        plt.xlim(min(all_areas[all_magnitudes>=m_filtering]), max(all_areas))
        plt.xlabel("Rupture area (m2)")
        plt.ylabel("Magnitude (Mw)")
        plt.grid("on", which="major", linestyle=":", zorder=0)
        plt.show()

        # Depth histograms

        thr_M = [6, 6.5, 7, 7.5, 8]
        labs = ["M6-6.4","M6.5-6.9","M7-7.4", "M>=7.5"]
        alphas = [0.2, 0.3, 0.6, 0.9]
        depths = np.arange(-20, 0, 1)
        for ms in range(len(thr_M)-1):
            loc_M = np.where((all_magnitudes>=thr_M[ms])&(all_magnitudes<thr_M[ms+1]))[0]
            num = all_z[loc_M]
            plt.hist(-all_z[loc_M], bins=depths, label = labs[ms], orientation="horizontal", density=False, alpha=alphas[ms], color="blue" )
            plt.legend()
        plt.xscale('log')
        plt.xlabel("Hypocenter count")
        plt.ylabel("Depth (km)")
        plt.show()

        for ms in range(len(thr_M)-1):
            loc_M = np.where((all_magnitudes>=thr_M[ms])&(all_magnitudes<thr_M[ms+1]))[0]
            num = centroid_z_re_all[loc_M]
            plt.hist(-centroid_z_re_all[loc_M], bins=depths, label = labs[ms], orientation="horizontal", density=False, alpha=alphas[ms], color="blue" )
            plt.legend()
        plt.xscale('log')
        plt.xlabel("Centroid count")
        plt.ylabel("Depth (km)")
        plt.show()




        # Plot Gutenberg Richter
        m_range_1 = np.arange(6, 8.1, 0.1)
        cum_rate = np.flip(np.cumsum(hist_after[::-1]/time_window))[np.where(mag_range==6)[0][0]]
        a = np.log10(cum_rate)+1*6
        a_real = np.log10(cum_rate) + abs(b) * 6
        N = 10**(a-1*m_range_1)
        N_actual = 10**(a_real-abs(b)*m_range_1)

        fig, ax1 = plt.subplots()

        ax1.scatter(mag_range[:-1], hist_after/time_window, c="grey")
        ax1.scatter(mag_range[:-1], np.flip(np.cumsum(hist_after[::-1]/time_window)), c="black")
        ax1.plot(mag_range[:-1], np.flip(np.cumsum(hist_after[::-1]/time_window)), c="black")
        ax1.plot(m_range_1, N, c="grey", linestyle="--", label="theoretical b=1")
        ax1.plot(m_range_1, N_actual, c="red", linestyle="--", label="real b="+str(round(b,2)))
        ax1.set_xlim(5, 8)
        ax1.set_yscale('log')
        plt.legend()
        plt.grid("on", which="major", linestyle=":", zorder=0)
        plt.xlabel("Magnitude (Mw)")
        plt.ylabel("Cumulative earthquake rate")
        plt.show()