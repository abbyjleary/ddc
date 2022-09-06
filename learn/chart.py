#!/usr/bin/python3

from collections import Counter
import random
from functools import reduce

import numpy as np

from beatcalc import BeatCalc
from util import make_onset_feature_context, np_pad


class Chart:
    def __init__(self, song_metadata, metadata, annotations):
        assert len(annotations) >= 2
        self.song_metadata = song_metadata
        self.metadata = metadata

        self.annotations = annotations
        self.label_counts = Counter()

        self.beat_calc = BeatCalc(song_metadata['offset'], song_metadata['bpms'], song_metadata['stops'])

        self.first_annotation_time = self.annotations[0][2]
        self.last_annotation_time = self.annotations[-1][2]

        assert self.last_annotation_time - self.first_annotation_time > 0.0

        self.time_annotated = self.last_annotation_time - self.first_annotation_time
        self.annotations_per_second = float(len(self.annotations)) / self.time_annotated

    def get_song_metadata(self):
        return self.song_metadata

    def get_coarse_difficulty(self):
        return self.metadata[0]

    def get_foot_difficulty(self):
        return self.metadata[1]

    def get_type(self):
        return self.metadata[2]

    def get_freetext(self):
        return self.metadata[3]

    def get_nannotations(self):
        return len(self.annotations)

    def get_time_annotated(self):
        return self.time_annotated

    def get_annotations_per_second(self):
        return self.annotations_per_second


class OnsetChart(Chart):
    def __init__(self, song_metadata, song_features, frame_rate, metadata, annotations):
        super(OnsetChart, self).__init__(song_metadata, metadata, annotations)

        self.song_features = song_features
        self.nframes = song_features.shape[0]

        self.dt = dt = 1.0 / frame_rate
        self.onsets = set(int(round(t / dt)) for _, _, t, _ in self.annotations)
        self.onsets = set(filter(lambda x: x >= 0, self.onsets))
        self.onsets = set(filter(lambda x: x < self.nframes, self.onsets))

        self.first_onset = min(self.onsets)
        self.last_onset = max(self.onsets)
        self.nframes_annotated = (self.last_onset - self.first_onset) + 1

        assert self.first_onset >= 0
        assert self.last_onset < self.nframes
        assert self.nframes > 0
        assert self.nframes_annotated > 0
        assert self.nframes >= self.nframes_annotated
        assert len(self.onsets) > 0

        self.blanks = set(range(self.nframes)) - self.onsets
        self._blanks_memoized = {}

    def get_nframes(self):
        return self.nframes

    def get_nframes_annotated(self):
        return self.nframes_annotated

    def get_nonsets(self):
        return len(self.onsets)

    def get_onsets(self):
        return self.onsets

    def get_first_onset(self):
        return self.first_onset

    def get_last_onset(self):
        return self.last_onset

    def get_example(self,
                    frame_idx,
                    dtype,
                    time_context_radius=1,
                    diff_feet_to_id=None,
                    diff_coarse_to_id=None,
                    diff_dipstick=False,
                    freetext_to_id=None,
                    beat_phase=False,
                    beat_phase_cos=False):
        feats_audio = make_onset_feature_context(self.song_features, frame_idx, time_context_radius)
        feats_other = [np.zeros(0, dtype=dtype)]

        if diff_feet_to_id:
            diff_feet = diff_feet_to_id[str(self.get_foot_difficulty())]
            diff_feet_onehot = np.zeros(max(diff_feet_to_id.values()) + 1, dtype=dtype)
            if diff_dipstick:
                diff_feet_onehot[:diff_feet + 1] = 1.0
            else:
                diff_feet_onehot[diff_feet] = 1.0
            feats_other.append(diff_feet_onehot)
        if diff_coarse_to_id:
            diff_coarse = diff_coarse_to_id[str(self.get_coarse_difficulty())]
            diff_coarse_onehot = np.zeros(max(diff_coarse_to_id.values()) + 1, dtype=dtype)
            if diff_dipstick:
                diff_coarse_onehot[:diff_coarse + 1] = 1.0
            else:
                diff_coarse_onehot[diff_coarse] = 1.0
            feats_other.append(diff_coarse_onehot)
        if freetext_to_id:
            freetext = self.get_freetext()
            if freetext not in freetext_to_id:
                freetext = None
            freetext_id = freetext_to_id[freetext]
            freetext_onehot = np.zeros(max(freetext_to_id.values()) + 1, dtype=dtype)
            freetext_onehot[freetext_id] = 1.0
            feats_other.append(freetext_onehot)
        if beat_phase:
            beat = self.beat_calc.time_to_beat(frame_idx * self.dt)
            beat_phase = beat - int(beat)
            feats_other.append(np.array([beat_phase], dtype=dtype))
        if beat_phase_cos:
            beat = self.beat_calc.time_to_beat(frame_idx * self.dt)
            beat_phase = beat - int(beat)
            beat_phase_cos = np.cos(beat * 2.0 * np.pi)
            beat_phase_cos += 1.0
            beat_phase_cos *= 0.5
            feats_other.append(np.array([beat_phase_cos], dtype=dtype))

        y = dtype(frame_idx in self.onsets)
        return np.array(feats_audio, dtype=dtype), np.concatenate(feats_other), y

    def sample(self, n, exclude_onset_neighbors=0, nunroll=0):
        if self._blanks_memoized:
            valid = self._blanks_memoized
        else:
            valid = set(range(self.get_first_onset(), self.get_last_onset() + 1))
            if exclude_onset_neighbors > 0:
                onset_neighbors = [set([x + i for x in self.onsets]) | set([x - i for x in self.onsets]) for i in range(
                    1, exclude_onset_neighbors + 1)]
                onset_neighbors = reduce(lambda x, y: x | y, onset_neighbors)
                valid -= onset_neighbors
            if nunroll > 0:
                valid -= set(range(self.get_first_onset(), self.get_first_onset() + nunroll))
            self._blanks_memoized = valid

        assert n <= len(valid)
        return random.sample(valid, n)

    def sample_onsets(self, n):
        assert n <= len(self.onsets)
        return random.sample(self.onsets, n)

    def sample_blanks(self, n, exclude_onset_neighbors=0, exclude_pre_onsets=True, exclude_post_onsets=True,
                      include_onsets=False):
        exclusion_params = (exclude_onset_neighbors, exclude_pre_onsets, exclude_post_onsets, include_onsets)

        if exclusion_params in self._blanks_memoized:
            blanks = self._blanks_memoized[exclusion_params]
        else:
            blanks = self.blanks
            if exclude_onset_neighbors > 0:
                onset_neighbors = [set([x + i for x in self.onsets]) | set([x - i for x in self.onsets]) for i in
                                   range(1, exclude_onset_neighbors + 1)]
                onset_neighbors = reduce(lambda x, y: x | y, onset_neighbors)
                blanks -= onset_neighbors
            if exclude_pre_onsets:
                blanks -= set(range(self.get_first_onset()))
            if exclude_post_onsets:
                blanks -= set(range(self.get_last_onset(), self.nframes))
            if include_onsets:
                blanks |= self.onsets
            self._blanks_memoized[exclusion_params] = blanks

        assert n <= len(blanks)
        return random.sample(blanks, n)

    def get_subsequence(self,
                        subseq_start,
                        subseq_len,
                        dtype,
                        zack_hack_div_2=0,
                        **feat_kwargs):
        seq_feats_audio = []
        seq_feats_other = []
        seq_y = []
        for i in range(subseq_start - zack_hack_div_2, subseq_start + subseq_len + zack_hack_div_2):
            feats_audio, feats_other, y = self.get_example(i, dtype=dtype, **feat_kwargs)
            seq_feats_audio.append(feats_audio)
            seq_feats_other.append(feats_other)
            seq_y.append(y)
        zhmin = zack_hack_div_2
        zhmax = zack_hack_div_2 + subseq_len

        return np.array(seq_feats_audio, dtype=dtype), np.array(seq_feats_other, dtype=dtype)
zhmin:zhmax], np.array(

    )