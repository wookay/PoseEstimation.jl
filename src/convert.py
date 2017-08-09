# Modifications from https://github.com/JustinShenk/video-pose-extractor


# From https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
# Based on @ZheC's repo
# Modifications by @JustinShenk

def caffenet(gpu):
    import caffe
    import os

    if gpu:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    else:
        caffe.set_mode_cpu()

    param, model = config_reader()
    deployFile = os.path.dirname(__file__) + '/' + model['deployFile']
    caffemodel = os.path.dirname(__file__) + '/' + model['caffemodel']
    net = caffe.Net(deployFile, caffemodel, caffe.TEST)
    return param, model, net

def prepare(gpu=False):
    return caffenet(gpu)

def netforward(param, model, tnet, image):
    import cv2 as cv
    import numpy as np
    import time

    # loop
    net = tnet
    enoughCount = 0
    LIMIT = 5
    scale_search = param['scale_search']
    boxsize = model['boxsize']
    heatmap_key = 'Mconv7_stage6_L2'
    paf_key = 'Mconv7_stage6_L1'
    if True:
        oriImg = cv.imread(image)
        #multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        multiplier = [0.5, 1]
        scale = multiplier[-1]
        imageToTest = cv.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(
            imageToTest, model['stride'], model['padValue'])
        net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
        net.blobs['data'].data[...] = np.transpose(np.float32(
            imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
        start_time = time.time()
        output_blobs = net.forward()
        print('At scale %f, The CNN took %.2f ms.' %
              (scale, 1000 * (time.time() - start_time)))

        blobdata = net.blobs[heatmap_key].data
        if blobdata.shape[1] != 19:
            print("err", blobdata.shape)

    # extract outputs, resize, and remove padding
    heatmap = np.transpose(np.squeeze(blobdata), (1, 2, 0))  # output 1 is heatmaps
    heatmap = cv.resize(heatmap, (0, 0), fx=model['stride'], fy=model[
                        'stride'], interpolation=cv.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0] -
                      pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    paf = np.transpose(np.squeeze(net.blobs[paf_key].data), (1, 2, 0))  # output 0 is PAFs
    paf = cv.resize(paf, (0, 0), fx=model['stride'], fy=model[
                    'stride'], interpolation=cv.INTER_CUBIC)
    paf = paf[:imageToTest_padded.shape[0] - pad[2],
              :imageToTest_padded.shape[1] - pad[3], :]
    paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    return oriImg, multiplier, heatmap, paf


def convert(currentFrame, oriImg, multiplier, heatmap, paf, jcolors, format='image'):
    import cv2 as cv
    from scipy.ndimage.filters import gaussian_filter
    import numpy as np
    import math
    import json

    thre1 = 0.1
    thre2 = 0.05

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    heatmap_avg = heatmap_avg + heatmap / len(multiplier)
    paf_avg = paf_avg + paf / len(multiplier)
    if True:
        # plt.imshow(heatmap_avg[:,:,2])
        all_peaks = []
        peak_counter = 0

        for part in range(19 - 1):
            x_list = []
            y_list = []
            map_ori = heatmap_avg[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre1))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(
                peaks_binary)[0]))  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[
                i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # find connection in the specified sequence, center 29 is in the position
        # 15
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                   [10, 11], [2, 12], [12, 13], [
                       13, 14], [2, 1], [1, 15], [15, 17],
                   [1, 16], [16, 18], [3, 17], [6, 18]]

        # the middle joints heatmap correpondence
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
                  [23, 24], [25, 26], [27, 28], [29, 30], [
                      47, 48], [49, 50], [53, 54], [51, 52],
                  [55, 56], [37, 38], [45, 46]]
        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if(nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        vec = np.divide(vec, norm)

                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(
                            vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        try:
                            score_with_dist_prior = sum(
                                score_midpts) / len(score_midpts) + min(0.5 * oriImg.shape[0] / norm - 1, 0)
                        except ZeroDivisionError:
                            print("Zero Division Error Encountered")
                            score_with_dist_prior = 0
                        criterion1 = len(np.nonzero(score_midpts > thre2)[
                                         0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(
                    connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if(i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack(
                            [connection, [candA[i][3], candB[j][3], s, i, j]])
                        if(len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall
        # configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array(
            [item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if(subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[
                                j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        print("found = 2")
                        membership = ((subset[j1] >= 0).astype(
                            int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[
                                j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i,
                                                                  :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        # visualize
        #colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
        #          [0, 255, 85], [0, 255, 170], [0, 255, 255], [
        #              0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        #          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        canvas = oriImg # cv.imread(image)

        # visualize 2
        stickwidth = 4

        body_parts = []
        for i in range(17):
            parts = {'id': i}
            subs = []
            for n in range(len(subset)):
                index = subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

                color = jcolors[i].tolist()
                #polygon = cv.ellipse2Poly((int(mY), int(mX)), (int(
                #    length / 2), stickwidth), int(angle), 0, 360, 1)
                #cv.polylines(cur_canvas, polygon, True, colors[i]) 

                polygon = cv.ellipse2Poly((int(mY), int(mX)), (int(
                    length / 2), stickwidth), int(angle), 0, 360, 1)
                cv.fillConvexPoly(cur_canvas, polygon, color)
                subinfo = {'sub_id': n, 'mX': mX, 'mY': mY, 'length': length, 'angle': angle}
                subs.append(subinfo)

                #positions[image].append((
                #    (int(mY), int(mX)),
                #    int(length / 2),
                #    int(angle), i))
                canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            parts = {'id': i, 'subset': subs}
            body_parts.append(parts)

        #plt.imshow(canvas[:, :, [2, 1, 0]])
        #fig = matplotlib.pyplot.gcf()
        #fig.set_size_inches(12, 12)
        info = {'body_parts': body_parts}

        if 'image' == format:
            outputPath = "output-%03d.png" % currentFrame
            cv.imwrite(outputPath, canvas)

        elif 'json' == format:
            outputPath = "output-%03d.json" % currentFrame
            with open(outputPath, 'w') as outfile:
                json.dump(info, outfile)

        print("saved frame to", outputPath)


def config_reader():
    from configobj import ConfigObj
    import os
    path = os.path.dirname(__file__) + '/' + 'config'
    #print("path", path)
    config = ConfigObj(path)

    param = config['param']
    model_id = param['modelID']
    model = config['models'][model_id]
    model['boxsize'] = int(model['boxsize'])
    model['stride'] = int(model['stride'])
    model['padValue'] = int(model['padValue'])
    #param['starting_range'] = float(param['starting_range'])
    #param['ending_range'] = float(param['ending_range'])
    param['octave'] = int(param['octave'])
    param['use_gpu'] = int(param['use_gpu'])
    param['starting_range'] = float(param['starting_range'])
    param['ending_range'] = float(param['ending_range'])
    param['scale_search'] = map(float, param['scale_search'])
    param['thre1'] = float(param['thre1'])
    param['thre2'] = float(param['thre2'])
    param['thre3'] = float(param['thre3'])
    param['mid_num'] = int(param['mid_num'])
    param['min_num'] = int(param['min_num'])
    param['crop_ratio'] = float(param['crop_ratio'])
    param['bbox_ratio'] = float(param['bbox_ratio'])
    param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])

    return param, model


def padRightDownCorner(img, stride, padValue):
    import numpy as np
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


#def load():
#    fld = h5py.File("heatmap_paf.hdf5", "r")
#    heatmap = fld['heatmap'][:]
#    paf = fld['paf'][:]
#    fld.close()
#    return heatmap, paf

#    f = h5py.File("heatmap_paf.hdf5", "w")
#    f.create_dataset('heatmap', data = heatmap)
#    f.create_dataset('paf', data = paf)
#    f.close()
#    print("written ")
