import numpy as np
from collections import deque
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as si
from scipy.ndimage.interpolation import zoom
import os
import sys
import copy
from scipy.ndimage.filters import gaussian_laplace
# from progress.bar import Bar

import cPickle as pickle


m = si.BrainObservatoryMonitor()
drive_path = '/data/dynamic-brain-workshop/brain_observatory_cache/'
manifest_file = os.path.join(drive_path, 'brain_observatory_manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_file)


sys.path.append('/home/michaelo/swdb_2017_tools/projects/deconvolution-inference/')
sys.path.append('/home/michaelo/swdb_2017_tools/projects/deconvolution-inference/OASIS')
import ca_tools as tools

##################################

# TODO:

# take vector of centers and crop
# whiten before/after average

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

        self.experiment_id = experiment_id
        self.downsample = downsample
        self._movie_names = ['natural_movie_one', 'natural_movie_two', 'natural_movie_three', 'natural_scenes']
        # self._movie_names = ['natural_movie_one', 'natural_movie_two', 'natural_movie_three']
        # self._movie_names = ['natural_scenes']

        self._movie_warps = {}

        for mn in self._movie_names:
            try:
                with open('/tmp/' + mn + '_' + str(self.downsample) + '.pickle', 'rb') as handle:
                    self._movie_warps[mn] = pickle.load(handle)
            except:
                continue

        # self._whitened_movie_warps = {}
        self._movie_sample_list = self._get_movie_sample_indexes(self.datasets)

        # calculate pixels per degree
        self.pixperdeg = 1 / m.pixels_to_visual_degrees(n=1)
        self.mask = m.get_mask()
        self.mask_min = [x.min() for x in np.where(self.mask)]
        self.mask_max = [x.max() for x in np.where(self.mask)]

        self._dffs = None
        self._pupil_locs = None
        self._shift_locs = None
        self._min_max_shift = None
        self._corrected_frame_numbers = None
        self._cell_indicies = None
        self._cell_ids = None

        try:
            with open('/tmp/' + str(hash(str(self.experiment_id) + str(self._movie_names))) + '.pickle', 'rb') as handle:
                self._events = pickle.load(handle)
        except:
            self._events = None

        self._STA = None
        self._MA = None

        self.chunk = 500

    @property
    def cell_ids(self):
        if self._cell_ids is None:
            cell_ids = set(self.datasets[0].get_cell_specimen_ids())
            for ds in self.datasets[1:]:
                cell_ids = cell_ids.intersection(set(ds.get_cell_specimen_ids()))
            self._cell_ids = list(cell_ids)
        return self._cell_ids

    @property
    def cell_indicies(self):
        if self._cell_indicies is None:
            self._cell_indicies = [ds.get_cell_specimen_indices(self.cell_ids) for ds in self.datasets]
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
                    frame_ends = list(stim_table['end'])
                    stim_range.append((frame_starts[0], np.maximum(frame_starts[-1] + 1, frame_ends[-1])))
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

    def warp_movie_to_screen(self, image, movie_name):
        if movie_name is 'natural_scenes':
            movie_warp = m.natural_scene_image_to_screen(image,  origin='upper')
        else:
            movie_warp = m.natural_movie_image_to_screen(image,  origin='upper')

        return zoom(movie_warp[self.mask_min[0]:self.mask_max[0], self.mask_min[1]:self.mask_max[1]], [self.downsample, self.downsample])

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

    def _make_shifted_stim_resp_generator(self, original_stim, shift_locations, frame_numbers, dff):
        '''
        make shifted stimuli

        '''
        chunk = self.chunk

        sh = original_stim.shape

        idx = range(0, len(frame_numbers), chunk)

        for cut in idx:

            sl = shift_locations[cut:cut+chunk]
            fn = frame_numbers[cut:cut+chunk]
            cdff = dff[:, cut:cut+chunk]
            # make larger stim defined by maximum shifts with a little extra slack
            shift_stim_shape = (len(sl), sh[1] + 2*np.maximum(self.min_max_shift[1][0], -self.min_max_shift[0][0]) + 3, sh[2] + 2*np.maximum(self.min_max_shift[1][1], -self.min_max_shift[0][1]) + 3)

            original_stim = (np.float32(original_stim)/255) - 0.5

            shift_stim = np.zeros(shift_stim_shape, dtype='float32')

            orig_stim = np.zeros((len(sl), original_stim.shape[1], original_stim.shape[2]), dtype='float32')

            sl = sl + [shift_stim_shape[1]/2, shift_stim_shape[2]/2]
            good_shift_locations = ~np.isnan(sl[:, 0])

            for i in range(len(sl)):
                if good_shift_locations[i]:
                    shift_stim[i, -sh[1]/2 + np.int32(sl[i, 0]):np.int32(sl[i, 0]) + sh[1]/2,
                                  -sh[2]/2 + np.int32(sl[i, 1]):np.int32(sl[i, 1]) + sh[2]/2] = original_stim[fn[i]]
                orig_stim[i] = original_stim[fn[i]]

            yield shift_stim, orig_stim, cdff

    def get_all_shifted_stims(self):
        all_shifted_stims = []
        for (ds, msl, sl, cfn) in zip(self.datasets, self._movie_sample_list, self.shift_locs, self.corrected_frame_numbers):
            shifted_stims = []
            for (movie_name, sl2, cfn2) in zip(msl[0], sl, cfn):
                if movie_name not in self._movie_warps.keys():
                    tmp_movie = self._get_stimulus_template(ds, movie_name)
                    tmp = self.warp_movie_to_screen(tmp_movie[0], movie_name)
                    tmp_warp = np.zeros((len(tmp_movie), tmp.shape[0], tmp.shape[1]), dtype='uint8')
                    for i in range(len(tmp)):
                        tmp_warp[i] = self.warp_movie_to_screen(tmp_movie[i], movie_name)
                    self._movie_warps[movie_name] = tmp_warp
                shifted_stims.append(self._make_shifted_stim(self._movie_warps[movie_name], sl2, cfn2))
            all_shifted_stims.append(shifted_stims)
        return all_shifted_stims

    def keras_generator(self, event_type='OASIS', delays=7, batch_size=400, shift=True):
        from keras.engine.training import _standardize_input_data
        if event_type not in self.events.keys():
            raise ValueError('Please specifiy one of the following for event_type: ' + str(self.events.keys()))

        movie_dict = self._movie_warps

        for (ds, msl, sl, cfn, dff, ci) in zip(self.datasets, self._movie_sample_list, self.shift_locs, self.corrected_frame_numbers, self.events[event_type], self.cell_indicies):
            for (movie_name, sl2, cfn2, dff2) in zip(msl[0], sl, cfn, dff):

                if movie_name not in movie_dict.keys():
                    tmp_movie = self._get_stimulus_template(ds, movie_name)
                    # bar = Bar('Processing ' + movie_name, max=len(tmp_movie))
                    tmp = self.warp_movie_to_screen(tmp_movie[0], movie_name)
                    tmp_warp = np.zeros((len(tmp_movie), tmp.shape[0], tmp.shape[1]), dtype='uint8')
                    for i in range(len(tmp_movie)):
                        tmp_warp[i] = self.warp_movie_to_screen(tmp_movie[i], movie_name)
                        # bar.next()

                    with open('/tmp/' + movie_name + '_' + str(self.downsample) + '.pickle', 'wb') as handle:
                        pickle.dump(tmp_warp, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    movie_dict[movie_name] = tmp_warp
                    # bar.finish()

                # ssg = self._make_shifted_stim_resp_generator(movie_dict[movie_name], sl2, cfn2, dff2)

                original_stim = movie_dict[movie_name]
                frame_numbers = cfn2
                shift_locations = sl2
                resp = dff2

                sh = original_stim.shape

                idx = range(0, len(frame_numbers), batch_size)
                # print(idx)

                for cut in idx:

                    sl3 = shift_locations[cut:cut+batch_size]
                    fn = frame_numbers[cut:cut+batch_size]
                    resp_out = resp[:, cut:cut+batch_size]
                    # make larger stim defined by maximum shifts with a little extra slack
                    shift_stim_shape = (len(sl3),
                                        sh[1] + 2*np.maximum(self.min_max_shift[1][0], -self.min_max_shift[0][0]) + 2,
                                        sh[2] + 2*np.maximum(self.min_max_shift[1][1], -self.min_max_shift[0][1]) + 2)

                    if shift:
                        out_stim = np.zeros(shift_stim_shape, dtype='float32')
                    else:
                        out_stim = np.zeros((len(sl3), original_stim.shape[1], original_stim.shape[2]), dtype='float32')

                    sl3 = sl3 + [shift_stim_shape[1]/2, shift_stim_shape[2]/2]
                    good_shift_locations = ~np.isnan(sl3[:, 0])

                    for i in range(len(sl3)):
                        if shift:
                            if good_shift_locations[i]:
                                out_stim[i, -sh[1]/2 + np.int32(sl3[i, 0]):np.int32(sl3[i, 0]) + sh[1]/2,
                                            -sh[2]/2 + np.int32(sl3[i, 1]):np.int32(sl3[i, 1]) + sh[2]/2] = original_stim[fn[i]]
                        else:
                            out_stim[i] = original_stim[fn[i]]

                    x = out_stim
                    batch_ids = np.arange(x.shape[0])
                    # print(batch_ids)

                    tlist = [1, 0] + list(range(2, np.ndim(x) + 1))

                    batch_ids = [np.maximum(0, batch_ids - d) for d in range(delays)]
                    x_batch = _standardize_input_data(x[batch_ids, :].transpose(tlist), ['x_batch'])

                    yield (x_batch, resp_out)

    def compute_STA(self, event_type='OASIS', delays=7, whiten=True, sigma=3, subtract_mean=True):

        if event_type not in self.events.keys():
            raise ValueError('Please specifiy one of the following for event_type: ' + str(self.events.keys()))

        if self._STA is None:
            STA = list(np.zeros(delays, dtype='float32'))
            STAc = list(np.zeros(delays, dtype='float32'))
            MA = list(np.zeros(delays, dtype='float32'))
            MAc = list(np.zeros(delays, dtype='float32'))
            count = list(np.zeros(delays, dtype='float32'))
            scount = list(np.zeros(delays, dtype='float32'))

            movie_dict = self._movie_warps

            for (ds, msl, sl, cfn, dff, ci) in zip(self.datasets, self._movie_sample_list, self.shift_locs, self.corrected_frame_numbers, self.events[event_type], self.cell_indicies):
                for (movie_name, sl2, cfn2, dff2) in zip(msl[0], sl, cfn, dff):

                    if movie_name not in movie_dict.keys():
                        tmp_movie = self._get_stimulus_template(ds, movie_name)

                        tmp = self.warp_movie_to_screen(tmp_movie[0], movie_name)
                        tmp_warp = np.zeros((len(tmp_movie), tmp.shape[0], tmp.shape[1]), dtype='uint8')
                        for i in range(len(tmp_movie)):
                            tmp_warp[i] = self.warp_movie_to_screen(tmp_movie[i], movie_name)

                        with open('/tmp/' + movie_name + '_' + str(self.downsample) + '.pickle', 'wb') as handle:
                            pickle.dump(tmp_warp, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        movie_dict[movie_name] = tmp_warp

                    ssg = self._make_shifted_stim_resp_generator(movie_dict[movie_name], sl2, cfn2, dff2)

                    for tmp_warp, orig, dff3 in ssg:
                        for d in range(delays):
                            STAc[d] += np.tensordot(dff3[:, d:], tmp_warp[:(len(tmp_warp)-d)][None, ...], axes=[1, 1])[:, 0]
                            STA[d] += np.tensordot(dff3[:, d:], orig[:(len(orig)-d)][None, ...], axes=[1, 1])[:, 0]
                            MAc[d] += np.sum(tmp_warp[:(len(tmp_warp)-d)], axis=0)
                            MA[d] += np.sum(orig[:(len(orig)-d)], axis=0)
                            scount[d] += np.sum(dff3[:, d:], axis=1)
                            count[d] += dff3[:, d:].shape[1]

            self._STAc = [mm/sc[:, None, None] for mm, sc in zip(STAc, scount)]
            self._STA = [mm/sc[:, None, None] for mm, sc in zip(STA, scount)]
            self._MA = [mm/cc for mm, cc in zip(MA, count)]
            self._MAc = [mm/cc for mm, cc in zip(MAc, count)]

        out_sta = copy.deepcopy(self._STA)
        out_stac = copy.deepcopy(self._STAc)

        if subtract_mean:
            out_sta = [x - y[None, ...] for x, y in zip(out_sta, self._MA)]
            out_stac = [x - y[None, ...] for x, y in zip(out_stac, self._MAc)]

        if whiten:
            for o in range(len(out_sta)):
                for on in range(out_sta[o].shape[0]):
                    out_sta[o][on] = -gaussian_laplace(out_sta[o][on], [sigma, sigma])
                    out_stac[o][on] = -gaussian_laplace(out_stac[o][on], [sigma, sigma])

        return out_sta, out_stac

    def _get_stimulus_template(self, dataset, stim_name):
        out = dataset.get_stimulus_template(stim_name)
        if stim_name is 'natural_scenes':
            out = np.vstack([128*np.ones((1, out.shape[1], out.shape[2]), dtype='uint8'), out])
        return out

    @property
    def dffs(self):
        if self._dffs is None:
            dffs = [dataset.get_dff_traces()[1][ci, :] * self.pixperdeg * self.downsample for dataset, ci in zip(self.datasets, self.cell_indicies)]

            dffs_list = []
            for (d, ms) in zip(dffs, self._movie_sample_list):
                dffs_list.append([d[:, mss[0]:mss[1]] for mss in ms[1]])

            self._dffs = dffs_list
        return self._dffs

    def _apply_deconvolution(self, dff):
        out = tools.ca_deconvolution(dff[0])

        for d in dff[1:]:
            tmp = tools.ca_deconvolution(d)
            for k in tmp.keys():
                out[k] = np.vstack([out[k], tmp[k]])
        return out

    @property
    def events(self):
        if self._events is None:
            dffs = [dataset.get_dff_traces()[1][ci, :] * self.pixperdeg * self.downsample for dataset, ci in zip(self.datasets, self.cell_indicies)]
            events = [self._apply_deconvolution(d) for d in dffs]

            keys = events[0].keys()
            events_dict = {}
            for k in keys:
                events_list = []
                for (e, ms) in zip(events, self._movie_sample_list):
                    events_list.append([e[k][:, mss[0]:mss[1]] for mss in ms[1]])
                events_dict[k] = events_list
            events_dict['dffs'] = self.dffs
            self._events = events_dict

            with open('/tmp/' + str(hash(str(self.experiment_id) + str(self._movie_names))) + '.pickle', 'wb') as handle:
                pickle.dump(events_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self._events

    @property
    def corrected_frame_numbers(self):
        if self._corrected_frame_numbers is None:
            corrected_frame_numbers = []
            for (dataset, ms) in zip(self.datasets, self._movie_sample_list):
                movie_cfn = []
                for movie_name in ms[0]:
                    stim_table = dataset.get_stimulus_table(movie_name)
                    frame_starts = deque(stim_table['start'])
                    frame_end = deque(stim_table['end']).pop()
                    if movie_name is 'natural_scenes':
                        frame_numbers = deque(stim_table['frame'] + 1)
                    else:
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

                    while frame_end > start_prev + 1:
                        cfn.append(cfn[-1])
                        start_prev += 1

                corrected_frame_numbers.append(movie_cfn)
            self._corrected_frame_numbers = corrected_frame_numbers
        return self._corrected_frame_numbers
