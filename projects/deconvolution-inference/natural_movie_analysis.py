import numpy as np
from collections import deque
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as si
from scipy.ndimage.interpolation import zoom
import os
import sys
from scipy.ndimage.filters import gaussian_laplace

import cPickle as pickle


m = si.BrainObservatoryMonitor()
drive_path = '/data/dynamic-brain-workshop/brain_observatory_cache/'
manifest_file = os.path.join(drive_path, 'brain_observatory_manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_file)


sys.path.append('~/swdb_2017_tools/projects/deconvolution-inference/OASIS')
import ca_tools as tools

##################################

# TODO:
# incorporate spikes
# incorporate natural images
# with/without eye correction
# take vector of centers and crop
# return cell specimen ids

##################################

def get_good_natural_movie_experiments(visual_area):
    movie_names = ['natural_movie_one', 'natural_movie_two', 'natural_movie_three']
    try:
        with open('/tmp/' + visual_area + '_eyetracking_data_dict.pickle', 'rb') as handle:
            good_exps = pickle.load(handle)
    except:
        exps = boc.get_experiment_containers(targeted_structures=[visual_area])
        good_exps = {}
        for exp in exps:
            expt_container_id = exp['id']
            session_id = [x['id'] for x in boc.get_ophys_experiments(experiment_container_ids=[expt_container_id], stimuli=movie_names)]
            not_failed = [not x['fail_eye_tracking'] for x in boc.get_ophys_experiments(experiment_container_ids=[expt_container_id], stimuli=movie_names, simple=False)]
            good_sess = [x[1] for x in zip(not_failed, session_id) if x[0]]
            if good_sess:
                good_exps[expt_container_id] = good_sess

        with open('/tmp/' + visual_area + '_eyetracking_data_dict.pickle', 'wb') as handle:
            pickle.dump(good_exps, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return good_exps


class natural_movie_analysis:
    def __init__(self, experiment_id, downsample=.25):

        self.visual_area = boc.get_ophys_experiments(experiment_container_ids=[experiment_id])[0]['targeted_structure']

        good_exps = get_good_natural_movie_experiments(self.visual_area)

        if experiment_id not in good_exps.keys():
            raise Exception('No eye-tracking for this experiment')

        sessions = good_exps[experiment_id]
        self.datasets = [boc.get_ophys_experiment_data(ophys_experiment_id=s) for s in sessions]

        self.downsample = downsample
        self._movie_names = ['natural_movie_one', 'natural_movie_two', 'natural_movie_three']
        self._movie_warps = {}
        self._whitened_movie_warps = {}
        self._movie_sample_list = self._get_movie_sample_indexes(self.datasets)

        # calculate pixels per degree
        self.pixperdeg = 1 / m.pixels_to_visual_degrees(n=1)

        self._dffs = None
        self._pupil_locs = None
        self._shift_locs = None
        self._min_max_shift = None
        self._corrected_frame_numbers = None
        self._cell_indicies = None

    @property
    def cell_indicies(self):
        if self._cell_indicies is None:
            cell_ids = set(self.datasets[0].get_cell_specimen_ids())
            for ds in self.datasets[1:]:
                cell_ids = cell_ids.intersection(set(ds.get_cell_specimen_ids()))
            cell_ids = list(cell_ids)
            self._cell_indicies = [ds.get_cell_specimen_indices(cell_ids) for ds in self.datasets]
        return self._cell_indicies

    def _get_movie_sample_indexes(self, datasets):
        movie_sample_list = []
        for dataset in datasets:
            movies_used = []
            stim_range = []

            for movie_name in self._movie_names:
                try:
                    stim_table = dataset.get_stimulus_table(movie_name)
                    frame_starts = list(stim_table['start'])
                    stim_range.append((frame_starts[0], frame_starts[-1] + 1))
                    movies_used.append(movie_name)
                except:
                    continue
            movie_sample_list.append([movies_used, stim_range])
        return movie_sample_list

    @property
    def pupil_locs(self):
        if self._pupil_locs is None:
            pupil_locs = [dataset.get_pupil_location()[1] * self.pixperdeg * self.downsample for dataset in self.datasets]

            pupil_loc_list = []
            for (sl, ms) in zip(pupil_locs, self._movie_sample_list):
                pupil_loc_list.append([sl[mss[0]:mss[1]] for mss in ms[1]])

            self._pupil_locs = pupil_loc_list
        return self._pupil_locs

    @property
    def shift_locs(self):
        if self._shift_locs is None:
            concat_pupil_locs = np.concatenate([np.concatenate(p, axis=0) for p in self.pupil_locs], axis=0)
            mean_loc = np.nanmean(concat_pupil_locs, axis=0)
            shift_locs = []
            for p in self.pupil_locs:
                shift_locs.append([(pp - mean_loc)[:, ::-1]*[1, -1] for pp in p])

            self._shift_locs = shift_locs
        return self._shift_locs

    @property
    def min_max_shift(self):
        if self._min_max_shift is None:
            concat_shift_locs = np.concatenate([np.concatenate(s, axis=0) for s in self.shift_locs], axis=0)
            self._min_max_shift = (np.int32(np.nanmin(concat_shift_locs, axis=0)), np.int32(np.nanmax(concat_shift_locs, axis=0)))

        return self._min_max_shift

    def warp_movie_to_screen(self, movie):
        tmp = m.natural_movie_image_to_screen(movie[0],  origin='upper')
        movie_warp = np.zeros((len(movie), tmp.shape[0], tmp.shape[1]), dtype='uint8')

        for i in range(len(movie)):
            movie_warp[i] = m.natural_movie_image_to_screen(movie[i], origin='upper')

        if self.downsample < 1:
            movie_warp = zoom(movie_warp, [1, self.downsample, self.downsample], order=1)

        return movie_warp

    def _make_shifted_stim(self, original_stim, shift_locations, frame_numbers):
        '''
        make shifted stimuli

        '''

        sh = original_stim.shape

        # make larger stim defined by maximum shifts with a little extra slack
        shift_stim_shape = (len(shift_locations), sh[1] + 2*np.maximum(self.min_max_shift[1][0], -self.min_max_shift[0][0]), sh[2] + 2*np.maximum(self.min_max_shift[1][1], -self.min_max_shift[0][1]))

        shift_stim = 128*np.ones(shift_stim_shape, dtype='uint8')

        shift_locations = shift_locations + [shift_stim_shape[1]/2, shift_stim_shape[2]/2]
        good_shift_locations = ~np.isnan(shift_locations[:, 0])

        for i in range(len(shift_locations)):
            if good_shift_locations[i]:
                shift_stim[i, -sh[1]/2 + np.int32(shift_locations[i, 0]):np.int32(shift_locations[i, 0]) + sh[1]/2,
                              -sh[2]/2 + np.int32(shift_locations[i, 1]):np.int32(shift_locations[i, 1]) + sh[2]/2] = original_stim[frame_numbers[i]]

        return shift_stim

    def _make_shifted_stim_resp_generator(self, original_stim, shift_locations, frame_numbers, dff, chunk=2000):
        '''
        make shifted stimuli

        '''

        sh = original_stim.shape

        idx = range(0, len(frame_numbers), chunk)

        for cut in idx:

            sl = shift_locations[cut:cut+chunk]
            fn = frame_numbers[cut:cut+chunk]
            cdff = dff[:, cut:cut+chunk]
            # make larger stim defined by maximum shifts with a little extra slack
            shift_stim_shape = (len(sl), sh[1] + 2*np.maximum(self.min_max_shift[1][0], -self.min_max_shift[0][0]) + 2, sh[2] + 2*np.maximum(self.min_max_shift[1][1], -self.min_max_shift[0][1]) + 2)

            shift_stim = 128*np.ones(shift_stim_shape, dtype='uint8')

            sl = sl + [shift_stim_shape[1]/2, shift_stim_shape[2]/2]
            good_shift_locations = ~np.isnan(sl[:, 0])

            for i in range(len(sl)):
                if good_shift_locations[i]:
                    shift_stim[i, -sh[1]/2 + np.int32(sl[i, 0]):np.int32(sl[i, 0]) + sh[1]/2,
                                  -sh[2]/2 + np.int32(sl[i, 1]):np.int32(sl[i, 1]) + sh[2]/2] = original_stim[fn[i]]

            yield shift_stim, cdff

    def get_all_shifted_stims(self):
        all_shifted_stims = []
        for (ds, msl, sl, cfn) in zip(self.datasets, self._movie_sample_list, self.shift_locs, self.corrected_frame_numbers):
            shifted_stims = []
            for (movie_name, sl2, cfn2) in zip(msl[0], sl, cfn):
                if movie_name not in self._movie_warps.keys():
                    tmp_movie = ds.get_stimulus_template(movie_name)
                    tmp = zoom(m.natural_movie_image_to_screen(tmp_movie[0], origin='upper'), [self.downsample, self.downsample], order=1)
                    tmp_warp = np.zeros((len(tmp_movie), tmp.shape[0], tmp.shape[1]), dtype='uint8')
                    for i in range(len(tmp)):
                        tmp_warp[i] = zoom(m.natural_movie_image_to_screen(tmp_movie[i], origin='upper'), [self.downsample, self.downsample], order=1)
                    self._movie_warps[movie_name] = tmp_warp
                shifted_stims.append(self._make_shifted_stim(self._movie_warps[movie_name], sl2, cfn2))
            all_shifted_stims.append(shifted_stims)
        return all_shifted_stims

    def compute_STA(self, delays=7, whiten=True, sigma=3):
        STA = list(np.zeros(delays, dtype='float32'))
        count = list(np.zeros(delays, dtype='float32'))

        if whiten:
            movie_dict = self._whitened_movie_warps
        else:
            movie_dict = self._movie_warps

        for (ds, msl, sl, cfn, dff, ci) in zip(self.datasets, self._movie_sample_list, self.shift_locs, self.corrected_frame_numbers, self.dffs, self.cell_indicies):
            for (movie_name, sl2, cfn2, dff2) in zip(msl[0], sl, cfn, dff):

                if movie_name not in movie_dict.keys():
                    tmp_movie = ds.get_stimulus_template(movie_name)
                    tmp = zoom(m.natural_movie_image_to_screen(tmp_movie[0], origin='upper'), [self.downsample, self.downsample], order=1)
                    tmp_warp = np.zeros((len(tmp_movie), tmp.shape[0], tmp.shape[1]), dtype='uint8')
                    for i in range(len(tmp)):
                        if whiten:
                            tw = -gaussian_laplace((np.float32(zoom(m.natural_movie_image_to_screen(tmp_movie[i], origin='upper'), [self.downsample, self.downsample], order=1)) / 255) - 0.5, sigma)
                            tw += .01
                            tw /= .025
                            tw *= 255
                            tmp_warp[i] = np.uint8(tw)
                        else:
                            tmp_warp[i] = zoom(m.natural_movie_image_to_screen(tmp_movie[i], origin='upper'), [self.downsample, self.downsample], order=1)
                    self._whitened_movie_warps[movie_name] = tmp_warp

                ssg = self._make_shifted_stim_resp_generator(movie_dict[movie_name], sl2, cfn2, dff2)

                for tmp_warp, dff3 in ssg:
                    tmp_warp = np.float32(tmp_warp) / 255
                    tmp_warp -= 0.5
                    for d in range(delays):
                        if d > 0:
                            STA[d] += np.tensordot(dff3[:, d:], tmp_warp[:-d][None, ...], axes=[1, 1])
                            count[d] += dff3[:, d:].shape[1]
                        else:
                            STA[d] += np.tensordot(dff3, tmp_warp[None, ...], axes=[1, 1])
                            count[d] += dff3.shape[1]

        return [mm/cc for mm, cc in zip(STA, count)]

    @property
    def dffs(self):
        if self._dffs is None:
            dffs = [dataset.get_dff_traces()[1][ci, :] * self.pixperdeg * self.downsample for dataset, ci in zip(self.datasets, self.cell_indicies)]

            dffs_list = []
            for (d, ms) in zip(dffs, self._movie_sample_list):
                dffs_list.append([d[:, mss[0]:mss[1]] for mss in ms[1]])

            self._dffs = dffs_list
        return self._dffs

    @property
    def events(self):
        if self._events is None:
            dffs = [dataset.get_dff_traces()[1][ci, :] * self.pixperdeg * self.downsample for dataset, ci in zip(self.datasets, self.cell_indicies)]
            

            out = [tools.ca_deconvolution(d) for d in dffs]
            
    @property
    def corrected_frame_numbers(self):
        if self._corrected_frame_numbers is None:
            corrected_frame_numbers = []
            for (dataset, ms) in zip(self.datasets, self._movie_sample_list):
                movie_cfn = []
                for movie_name in ms[0]:
                    stim_table = dataset.get_stimulus_table(movie_name)
                    frame_starts = deque(stim_table['start'])
                    frame_numbers = deque(stim_table['frame'])

                    cfn = [frame_numbers.popleft()]
                    start_prev = frame_starts.popleft()

                    while frame_starts:
                        start = frame_starts.popleft()
                        # repeat previous frame number if next frame starts later
                        while start > start_prev + 1:
                            cfn.append(cfn[-1])
                            start_prev += 1
                        start_prev += 1
                        cfn.append(frame_numbers.popleft())
                    movie_cfn.append(cfn)

                corrected_frame_numbers.append(movie_cfn)
            self._corrected_frame_numbers = corrected_frame_numbers
        return self._corrected_frame_numbers
