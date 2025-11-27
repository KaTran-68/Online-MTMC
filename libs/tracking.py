################################
#         Spatial Tracking     #
################################

import numpy as np
import numpy.matlib
import math
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from thirdparty import bbox
import torch
import matplotlib.pyplot as plt

class tracking():

    def __init__(self, mtmc, CONFIG):
        self.tracks_KF = []
        self.id_track = 1
        self.unmatched_tracks, self.unmatched_clusters = [], []
        self.matches = None
        self.updated_flag = 0
        self.CONFIG = CONFIG

    def new_track(self, id, centroid, kalman, cluster_id):
        # fromcluster lưu danh sách id cluster đã gán trong frame hiện tại (thường 1, đôi khi >1 nếu merge)
        track = {
            'id': id,
            'xw': float(np.asarray(centroid[0]).reshape(-1)[0]),
            'yw': float(np.asarray(centroid[1]).reshape(-1)[0]),
            'kalmanFilter': kalman,
            'age': 1,
            'state': 0,
            'totalVisibleCount': 1,
            'consecutiveInvisibleCount': 0,
            'fromcluster': [int(cluster_id)]  # list thay vì np.array
        }
        return track

    def new_global_track(self):
        return {'id': [], 'xw': [], 'yw': [], 'det': []}

    def create_new_tracks_KF(self, clusters):
    # Debug-safe version: prints internal state and guards exceptions
        try:
            print("DEBUG create_new_tracks_KF: unmatched_clusters =", self.unmatched_clusters)
            if not isinstance(clusters, (list, tuple)):
                print("DEBUG create_new_tracks_KF: clusters is not a list/tuple:", type(clusters))
                return

            # Build centroids for unmatched clusters
            centroids_xw = [clusters[item]['xw'] for item in self.unmatched_clusters] if len(self.unmatched_clusters) > 0 else []
            centroids_yw = [clusters[item]['yw'] for item in self.unmatched_clusters] if len(self.unmatched_clusters) > 0 else []

            num_centroids = len(centroids_xw)
            print(f"DEBUG create_new_tracks_KF: num_centroids = {num_centroids}")

            centroids = np.zeros((num_centroids, 2))
            if num_centroids > 0:
                centroids[:, 0] = centroids_xw
                centroids[:, 1] = centroids_yw

            for idx_in_unmatched in range(len(self.unmatched_clusters)):
                try:
                    centroid = centroids[idx_in_unmatched, :]
                except Exception as e:
                    print("DEBUG create_new_tracks_KF: failed to read centroid at", idx_in_unmatched, "error:", e)
                    continue

                cluster_idx = int(self.unmatched_clusters[idx_in_unmatched])
                # quick sanity: cluster should exist
                if cluster_idx < 0 or cluster_idx >= len(clusters):
                    print("DEBUG create_new_tracks_KF: cluster_idx out of range:", cluster_idx)
                    continue

                # Try to append (no reuse logic here) — keep this simple for debug
                try:
                    kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
                    dt = 1.
                    kalman_filter.F = np.array([[1, dt, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, dt],
                                                [0, 0, 0, 1]])
                    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)
                    kalman_filter.Q = block_diag(q, q)
                    kalman_filter.x = np.array([[centroid[0], 0, centroid[1], 0]]).T
                    kalman_filter.H = np.array([[1, 0, 0, 0],
                                                [0, 0, 1, 0]])
                    kalman_filter.R = np.array([[5, 0],
                                                [0, 5]])
                    kalman_filter.P *= 500.

                    new_trk = self.new_track(self.id_track, centroid, kalman_filter, cluster_idx)
                    # debug: check new_trk shape/keys
                    print(f"DEBUG create_new_tracks_KF: creating track id={self.id_track}, centroid={centroid}, cluster_idx={cluster_idx}")
                    self.tracks_KF.append(new_trk)
                    self.id_track += 1
                except Exception as e:
                    print("DEBUG create_new_tracks_KF: error while creating appending track for cluster", cluster_idx, "error:", e)
                    continue

            print("DEBUG create_new_tracks_KF: after creation, num tracks =", len(self.tracks_KF))
        except Exception as e_outer:
            print("DEBUG create_new_tracks_KF: outer exception:", e_outer)
            # don't crash the main loop
            return

    def predict_new_locations(self):
        for trk in self.tracks_KF:
            trk['kalmanFilter'].predict()
            x = trk['kalmanFilter'].x
            # Ép về float scalar tránh (N,1) hoặc (1,) len-1 array
            trk['xw'] = float(x[0, 0]) if x.ndim == 2 else float(x[0])
            trk['yw'] = float(x[2, 0]) if x.ndim == 2 else float(x[2])

    def assign_detections_to_tracks(self, cost, clusters, pos_track_HA, ids_track_HA, tracks_id,
                                    pos_track_to_remove, pos_cluster_HA, ids_cluster_HA, clusters_id,
                                    pos_cluster_to_remove):
        self.unmatched_tracks, self.unmatched_clusters = [], []

        # linear_sum_assignment trả về hai mảng (row_ind, col_ind)
        row_ind, col_ind = linear_sum_assignment(cost)
        self.matches = [np.asarray(row_ind, dtype=np.int64), np.asarray(col_ind, dtype=np.int64)]

        # tracks unassigned / clusters unassigned wrt the HA (high-association) indexes
        tracks_unassigned_HA = []
        clusters_unassigned_HA = []

        for t in pos_track_HA:
            if t not in self.matches[0]:
                tracks_unassigned_HA.append(np.int64(t))

        for c in pos_cluster_HA:
            if c not in self.matches[1]:
                clusters_unassigned_HA.append(np.int64(c))

        # Update clusters: map matched HA indices back to real cluster positions
        clusters_assigned_HA = self.matches[1]
        ids_clusters_assigned_HA = ids_cluster_HA[clusters_assigned_HA]
        # produce array of real positions (one per matched HA index)
        real_pos_clusters_matched = []
        for cid in ids_clusters_assigned_HA:
            idxs = np.where(clusters_id == cid)[0]
            if idxs.size > 0:
                real_pos_clusters_matched.append(np.int64(idxs[0]))
        self.matches[1] = np.asarray(real_pos_clusters_matched, dtype=np.int64)

        # Unassigned clusters: map and collect first match if exists
        ids_clusters_unassigned_HA = ids_cluster_HA[clusters_unassigned_HA]
        real_pos_clusters_unmatched = []
        for cid in ids_clusters_unassigned_HA:
            idxs = np.where(clusters_id == cid)[0]
            if idxs.size > 0:
                real_pos_clusters_unmatched.append(np.int64(idxs[0]))
        self.unmatched_clusters = list(real_pos_clusters_unmatched)

        # Update tracks: map matched HA indices back to real track positions
        tracks_assigned_HA = self.matches[0]
        ids_tracks_assigned_HA = ids_track_HA[tracks_assigned_HA]
        real_pos_tracks_matched = []
        for tid in ids_tracks_assigned_HA:
            idxs = np.where(tracks_id == tid)[0]
            if idxs.size > 0:
                real_pos_tracks_matched.append(np.int64(idxs[0]))
        self.matches[0] = np.asarray(real_pos_tracks_matched, dtype=np.int64)

        # Unassigned tracks: map and collect
        ids_tracks_unassigned_HA = ids_track_HA[tracks_unassigned_HA]
        real_pos_tracks_unmatched = []
        for tid in ids_tracks_unassigned_HA:
            idxs = np.where(tracks_id == tid)[0]
            if idxs.size > 0:
                real_pos_tracks_unmatched.append(np.int64(idxs[0]))
        self.unmatched_tracks = list(real_pos_tracks_unmatched)

        # Add filtered removals
        for i in pos_track_to_remove:
            self.unmatched_tracks.append(np.int64(i))
        for i in pos_cluster_to_remove:
            self.unmatched_clusters.append(np.int64(i))


    def cluster_track_assignment(self, clusters, display):
        # Ép về float 1D để tránh shape (N,1)
        clusters_xw = np.array([float(np.asarray(item['xw']).reshape(-1)[0]) for item in clusters], dtype=float)
        clusters_yw = np.array([float(np.asarray(item['yw']).reshape(-1)[0]) for item in clusters], dtype=float)
        num_clusters = clusters_xw.size

        clusters_position = np.zeros((num_clusters, 2), dtype=float)
        if num_clusters > 0:
            clusters_position[:, 0] = clusters_xw
            clusters_position[:, 1] = clusters_yw
        clusters_id = np.arange(num_clusters)

        tracks_xw = np.array([float(np.asarray(item['xw']).reshape(-1)[0]) for item in self.tracks_KF], dtype=float)
        tracks_yw = np.array([float(np.asarray(item['yw']).reshape(-1)[0]) for item in self.tracks_KF], dtype=float)
        tracks_id = np.array([item['id'] for item in self.tracks_KF])
        num_tracks = tracks_xw.size

        tracks_position = np.zeros((num_tracks, 2), dtype=float)
        if num_tracks > 0:
            tracks_position[:, 0] = tracks_xw
            tracks_position[:, 1] = tracks_yw

        cost = np.zeros((num_tracks, num_clusters), dtype=float)

        if num_clusters != 0 and num_tracks != 0:
            for i in range(num_tracks):
                cost[i, :] = np.linalg.norm(clusters_position - np.matlib.repmat(tracks_position[i, :], num_clusters, 1), axis=1)

            pos_track_to_remove = np.where(np.sum((cost < self.CONFIG['DIST_TH']) * 1, axis=1) == 0)[0]
            ids_track_HA = np.delete(tracks_id, pos_track_to_remove)
            pos_track_HA = np.arange(num_tracks - len(pos_track_to_remove))
            cost_filtered = np.delete(cost, pos_track_to_remove, axis=0)

            pos_cluster_to_remove = np.where(np.sum((cost < (self.CONFIG['DIST_TH'] + 1e-5)) * 1, axis=0) == 0)[0]
            ids_cluster_HA = np.delete(clusters_id, pos_cluster_to_remove)
            pos_cluster_HA = np.arange(num_clusters - len(pos_cluster_to_remove))
            cost_filtered = np.delete(cost_filtered, pos_cluster_to_remove, axis=1)

            self.assign_detections_to_tracks(cost_filtered, clusters, pos_track_HA, ids_track_HA, tracks_id,
                                             pos_track_to_remove, pos_cluster_HA, ids_cluster_HA, clusters_id,
                                             pos_cluster_to_remove)
        else:
            # đảm bảo matches luôn là hai mảng 1-D (rỗng nếu không có)
            self.matches = [np.array([], dtype=np.int64), np.array([], dtype=np.int64)]
            if num_clusters == 0:
                self.unmatched_clusters = []
                self.unmatched_tracks = list(np.arange(num_tracks, dtype=np.int64))
            if num_tracks == 0:
                self.unmatched_tracks = []
                self.unmatched_clusters = list(np.arange(num_clusters, dtype=np.int64))


    def update_assigned_tracks(self, clusters):
        # đảm bảo self.matches[0] luôn là array (rỗng nếu không có)
        if not self.matches or not isinstance(self.matches, (list, tuple)):
            num_matched_tracks = 0
        else:
            m0 = self.matches[0]
            # nếu scalar, coi là 1 match
            if not hasattr(m0, '__len__'):
                num_matched_tracks = 1
                m0 = np.asarray([m0], dtype=np.int64)
                self.matches[0] = m0
            else:
                num_matched_tracks = len(m0)

        for i in range(num_matched_tracks):
            track_id = int(self.matches[0][i])
            cluster_id = int(self.matches[1][i])
            z = np.array([clusters[cluster_id]['xw'], clusters[cluster_id]['yw']], dtype=float)
            self.tracks_KF[track_id]['kalmanFilter'].update(z)
            # Cập nhật lại xw, yw dạng float sau update
            x = self.tracks_KF[track_id]['kalmanFilter'].x
            self.tracks_KF[track_id]['xw'] = float(x[0, 0]) if x.ndim == 2 else float(x[0])
            self.tracks_KF[track_id]['yw'] = float(x[2, 0]) if x.ndim == 2 else float(x[2])

            if (self.tracks_KF[track_id]['totalVisibleCount'] >= 3 and
                    self.tracks_KF[track_id]['consecutiveInvisibleCount'] == 0):
                self.tracks_KF[track_id]['state'] = 1
            self.tracks_KF[track_id]['age'] += 1
            self.tracks_KF[track_id]['totalVisibleCount'] += 1
            self.tracks_KF[track_id]['consecutiveInvisibleCount'] = 0
            self.tracks_KF[track_id]['fromcluster'] = [int(cluster_id)]

    def update_unassigned_tracks(self):
        for track_id in self.unmatched_tracks:
            self.tracks_KF[track_id]['consecutiveInvisibleCount'] += 1
            self.tracks_KF[track_id]['age'] += 1
            self.tracks_KF[track_id]['fromcluster'] = []  # empty list

    def check_unassigned_clusters(self, clusters, association_matrix, dist_features, dist_spatial):
        if (len(self.unmatched_clusters) == 0 or
            len(self.tracks_KF) == 0 or
            association_matrix is None or
            association_matrix.size == 0):
            return

        # Chuẩn hoá dist_features
        df = None
        if isinstance(dist_features, np.ndarray) and dist_features.ndim == 2:
            df = dist_features  # NxN khoảng cách giữa detections

        unmatched_copy = self.unmatched_clusters.copy()

        for u_cl in unmatched_copy:
            dets_in_unmatched = clusters[u_cl]['det']
            if len(dets_in_unmatched) != 1:
                continue

            detection_self = dets_in_unmatched[0]['id_global']
            if detection_self >= association_matrix.shape[0]:
                continue

            valid_matches = np.where(association_matrix[detection_self, :] == 1)[0]
            if valid_matches.size == 0:
                continue

            if df is not None:
                if detection_self >= df.shape[0]:
                    continue
                posible_features_row = df[detection_self, valid_matches]
            else:
                if (isinstance(dist_spatial, np.ndarray) and
                        dist_spatial.ndim == 2 and
                        u_cl < dist_spatial.shape[0]):
                    posible_features_row = dist_spatial[u_cl, valid_matches % dist_spatial.shape[1]]
                else:
                    continue

            if posible_features_row.size == 0:
                continue

            min_pos = int(np.argmin(posible_features_row))
            detection_to_join = valid_matches[min_pos]

            cluster_to_join = None
            for cl_idx, cl in enumerate(clusters):
                ids_in_cl = [d['id_global'] for d in cl['det']]
                if detection_to_join in ids_in_cl:
                    if detection_self < association_matrix.shape[0]:
                        block_vals = association_matrix[detection_self, ids_in_cl]
                        if 100 in block_vals:
                            continue
                    cluster_to_join = cl_idx
                    break

            if cluster_to_join is None:
                continue

            for trk in self.tracks_KF:
                if cluster_to_join in trk['fromcluster']:
                    if u_cl not in trk['fromcluster']:
                        trk['fromcluster'].append(int(u_cl))
                    if u_cl in self.unmatched_clusters:
                        self.unmatched_clusters.remove(u_cl)
                    break

    def delete_lost_tracks(self):
        invisible_for_too_long = 10 if self.CONFIG['BLIND_OCCLUSION'] else 2
        age_threshold = 8
        ages = np.array([trk['age'] for trk in self.tracks_KF])
        if ages.size != 0:
            totalVisibleCounts = np.array([trk['totalVisibleCount'] for trk in self.tracks_KF])
            visibility = totalVisibleCounts / ages
            consecutiveInvisibleCount = np.array([trk['consecutiveInvisibleCount'] for trk in self.tracks_KF])
            lostInds = np.bitwise_or(np.bitwise_and(ages < age_threshold, visibility < 0.5),
                                     consecutiveInvisibleCount >= invisible_for_too_long)
            self.tracks_KF = [trk for idx, trk in enumerate(self.tracks_KF) if not lostInds[idx]]

    def save_global_tracking_data(self, clusters, f, global_tracks, cam):
        num_tracks = len(self.tracks_KF)
        if self.CONFIG['BLIND_OCCLUSION']:
            invisible = 2
        else:
            invisible = 1

        if num_tracks != 0:
            self.updated_flag = 1
            for i in range(num_tracks):
                trk = self.tracks_KF[i]
                if trk['consecutiveInvisibleCount'] < invisible:
                    global_tracks[f].append(self.new_global_track())
                    gtrk = global_tracks[f][-1]
                    gtrk['id'] = trk['id']
                    gtrk['xw'] = float(trk['xw'])
                    gtrk['yw'] = float(trk['yw'])
                    from_cluster = trk['fromcluster']

                    for cid in from_cluster:
                        for det in clusters[cid]['det']:
                            gtrk['det'].append(det)

                    if self.CONFIG.get('REPROJECTION', False):
                        if len(from_cluster) == 0 and f > 0 and len(global_tracks[f-1]) > 0:
                            prev_candidates = [item for item in global_tracks[f-1] if item['id'] == gtrk['id']]
                            if prev_candidates:
                                gtrk['det'] = prev_candidates[0]['det']
                                prev_bbox = gtrk['det'][0]
                                centroid_x = trk['xw']
                                centroid_y = trk['yw']
                                cam_key = 'c00' + str(prev_bbox['id_cam'])
                                if cam_key in cam.homography_matrix:
                                    base_x, base_y = cam.apply_homography_world_to_image(
                                        float(centroid_x), float(-centroid_y), cam.homography_matrix[cam_key]
                                    )
                                    x = int(round(base_x - (prev_bbox['w'] / 2)))
                                    y = int(round(base_y - prev_bbox['h']))
                                    gtrk['det'][0]['x'] = x
                                    gtrk['det'][0]['y'] = y

            # Clean overlapping same-camera tracks
            dets = []
            ids_cam = []
            ids_track = []
            id_tracks_to_clean = []

            for item in global_tracks[f]:
                for d in item['det']:
                    dets.append([d['x'], d['y'], d['w'], d['h']])
                    ids_cam.append(d['id_cam'])
                ids_track.extend([item['id']] * len(item['det']))

            for i in range(len(dets)):
                for j in range(i + 1, len(dets)):
                    det1 = dets[i]
                    det2 = dets[j]
                    box1 = np.array((det1[0], det1[1], det1[0] + det1[2], det1[1] + det1[3]))
                    box2 = np.array((det2[0], det2[1], det2[0] + det2[2], det2[1] + det2[3]))
                    iou = bbox.bbox_iou(torch.from_numpy(box1).cuda(), torch.from_numpy(box2).cuda())
                    if iou.item() > 0.8 and ids_cam[i] == ids_cam[j]:
                        id_track1 = ids_track[i]
                        id_track2 = ids_track[j]
                        age1 = [t['age'] for t in self.tracks_KF if t['id'] == id_track1][0]
                        age2 = [t['age'] for t in self.tracks_KF if t['id'] == id_track2][0]
                        if age1 > age2:
                            id_tracks_to_clean.append(id_track2)
                        else:
                            id_tracks_to_clean.append(id_track1)

            if id_tracks_to_clean:
                self.tracks_KF = [t for t in self.tracks_KF if t['id'] not in id_tracks_to_clean]
                global_tracks[f] = [g for g in global_tracks[f] if g['id'] not in id_tracks_to_clean]
        else:
            self.updated_flag = 0
            global_tracks[f] = []

        return global_tracks

    def display_tracks(self):
        if len(self.tracks_KF) > 0:
            for t in self.tracks_KF:
                plt.plot(t['xw'], t['yw'], '*', lineWidth=1, markerSize=10, color='red')
                plt.text(t['xw'] - 5e-6, t['yw'] + 5e-6, str(t['id']), fontsize=15, color='red')