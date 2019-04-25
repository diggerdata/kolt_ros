from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Track(object):
    def __init__(self, prediction, trackIdCount, rate, ra=1.5, sv=3.0):
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter(rate=rate, ra=ra, sv=sv)  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y,z)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path


class Tracker(object):
    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, track_id_count, rate, ra=1.5, sv=3.0):
        self.rate = rate
        self.sv = sv
        self.ra = ra
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length + 1
        self.tracks = []
        self.track_id_count = track_id_count

    def update(self, detections):
        # TODO: Simplify whole thing

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i].T, self.track_id_count, self.rate, ra=self.ra, sv=self.sv)
                self.track_id_count += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            try:
                diff = detections.T - self.tracks[i].prediction
                dist = np.sqrt(np.sum(diff**2, axis=1)).T
                cost[i,:] = dist
            except:
                continue
        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        # TODO: Find a way to vectorize this:
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]].T,
                              self.track_id_count, self.rate)
                self.track_id_count += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()
            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            detections[assignment[i]].reshape((3,1)), True)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            np.zeros((3,1)), False)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction
        return self._tracks_to_poses()
        # return self.tracks

    def _tracks_to_poses(self):
        predictions = []
        traces = {}
        for track in self.tracks:
            predictions.append(track.prediction)
            traces[track.track_id] = track.trace
        return predictions, traces