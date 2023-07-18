import numpy as np
from collections import defaultdict
import os, cv2, json


class HICOEvaluator():
    def __init__(self, preds, gts, rare_triplets, non_rare_triplets, correct_mat, args):
        self.overlap_iou = 0.5
        self.max_hois = 100 # 300, 100, 64, 32
        #self.max_hois = 300 # 300, 100, 64, 32

        self.use_nms_filter = args.use_nms_filter
        #self.thres_nms = 0.83  # 0.9, 0.87, 0.85, 0.83, 0.8 for mode 2
        self.thres_nms = args.thres_nms  # 0.9, 0.87, 0.85, 0.83, 0.8 for mode 2
        #self.thres_nms = 0.95  # 0.9, 0.87, 0.85, 0.83, 0.8 for mode 1
        self.nms_alpha = args.nms_alpha
        self.nms_beta = args.nms_beta

        self.use_score_thres = False
        self.thres_score = 1e-5  #  5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4

        self.use_soft_nms = False
        self.soft_nms_sigma = 0.5
        self.soft_nms_thres_score = 1e-11 # 5e-6, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11


        self.rare_triplets = rare_triplets
        self.non_rare_triplets = non_rare_triplets

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_triplets = []

        #print('correct_mat: ', correct_mat, correct_mat.shape)
        self.preds = []
        for index, img_preds in enumerate(preds):
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': list(bbox), 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']
            #print(hoi_scores.shape)
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            #print(verb_labels.shape)
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            #print(subject_ids.shape)
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()
            #print(hoi_scores.shape)
            #print(verb_labels.shape)
            #print(subject_ids.shape)

            if len(subject_ids) > 0:
                object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.max_hois]
            else:
                hois = []

            filename = gts[index]['filename']
            self.preds.append({
                'filename':filename,
                'predictions': bboxes,
                'hoi_prediction': hois
            })


        if self.use_nms_filter:
            print('eval use_nms_filter ...')
            self.preds = self.triplet_nms_filter(self.preds)

        ### clip boxes
        #self.preds = self.clip_preds_boxes(self.preds)

        self.gts = []
        for i, img_gts in enumerate(gts):
            filename = img_gts['filename']
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k != 'id' and k != 'filename'}
            self.gts.append({
                'filename':filename,
                'annotations': [{'bbox': list(bbox), 'category_id': label} for bbox, label in zip(img_gts['boxes'], img_gts['labels'])],
                'hoi_annotation': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': 1} for hoi in img_gts['hois']]
            })
            for hoi in self.gts[-1]['hoi_annotation']:
                triplet = (self.gts[-1]['annotations'][hoi['subject_id']]['category_id'],
                           self.gts[-1]['annotations'][hoi['object_id']]['category_id'],
                           1)
                           #hoi['category_id'])

                if triplet not in self.gt_triplets:
                    self.gt_triplets.append(triplet)

                self.sum_gts[triplet] += 1


        with open(args.json_file, 'w') as f:
            f.write(json.dumps(str({'preds':self.preds, 'gts':self.gts})))




        self.use_json = True
        #self.use_json = False
        print('use_json: ', self.use_json)
        if self.use_json:
            self.sum_gts = defaultdict(lambda: 0)
            self.gt_triplets = []

            #with open('results_json_table_a/results_6v6_q64_stage2_top100.json') as f:  #  CDN-S
            #with open('results_json_table_a/results_6v6_q64_stage2_top100_binary.json') as f:  #  CDN-S
            #with open('results_json_table_a/results_ho-pd_binary.json') as f: 
            #with open('results_json_table_a/results_ican_detr_binary.json') as f: 
            #with open('results_json_table_a/results_ican_binary.json') as f: 
            with open('results_json_table_a/results_ho-pd_only_binary.json') as f: 
                results = eval(json.loads(f.read()))

                self.preds = []
                for temp in results['preds']:
                    #if temp['filename'] != 'HICO_test2015_00001193.jpg': continue   # TODO
                    self.preds.append( {'filename': temp['filename'],
                                        'predictions': temp['predictions'],
                                        'hoi_prediction': temp['hoi_prediction'][:self.max_hois] })

                self.gts = []
                for temp in results['gts']:
                    #if temp['filename'] != 'HICO_test2015_00001193.jpg': continue   # TODO
                    self.gts.append( {'filename': temp['filename'],
                                        'annotations': temp['annotations'],
                                        'hoi_annotation': temp['hoi_annotation'] })

                    for hoi in temp['hoi_annotation']:
                        triplet = (temp['annotations'][hoi['subject_id']]['category_id'],
                                   temp['annotations'][hoi['object_id']]['category_id'],
                                   1)
                                   #hoi['category_id'])
        
                        if triplet not in self.gt_triplets:
                            self.gt_triplets.append(triplet)
        
                        self.sum_gts[triplet] += 1

                #if use_nms_filter:
                #    print('eval use_nms_filter ...')
                #    self.preds = self.triplet_nms_filter(self.preds)

        print(len(self.preds))
        print(len(self.gts))


    def evaluate(self):
        for img_preds, img_gts in zip(self.preds, self.gts):
            if img_preds['filename'] == 'HICO_test2015_00001193.jpg':
                print(img_preds)
                print(img_gts)
            pred_bboxes = img_preds['predictions']
            if len(pred_bboxes) == 0: continue

            gt_bboxes = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            #print('pred_bboxes: ', pred_bboxes, len(pred_bboxes))
            #print('pred_hois: ', pred_hois, len(pred_hois))
            #print('gt_hois: ', gt_hois, len(gt_hois))
            #print('pred_hois: ', len(pred_hois), 'gt_hois: ', len(gt_hois))
            if len(gt_bboxes) != 0:
                #print(gt_bboxes)
                #print(pred_bboxes)
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_hoi in pred_hois:
                    triplet = [pred_bboxes[pred_hoi['subject_id']]['category_id'],
                               pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id']]
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_hoi['score'])
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                if triplet in self.rare_triplets:
                    rare_ap[triplet] = 0
                elif triplet in self.non_rare_triplets:
                    non_rare_ap[triplet] = 0
                else:
                    print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
                continue

            score = np.array(self.score[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            #ap[triplet] = self.cal_prec(rec, prec)
            ap[triplet] = self.voc_ap(rec, prec)
            max_recall[triplet] = np.amax(rec)
            if triplet in self.rare_triplets:
                rare_ap[triplet] = ap[triplet]
            elif triplet in self.non_rare_triplets:
                non_rare_ap[triplet] = ap[triplet]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
        m_ap = np.mean(list(ap.values()))
        m_ap_rare = np.mean(list(rare_ap.values()))
        m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))

        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare, m_max_recall))
        print('--------------------')

        return {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare, 'mean max recall': m_max_recall}

    def cal_prec(self, rec, prec, t=0.8):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        return p

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hois, gt_hois, match_pairs, pred_bboxes, bbox_overlaps):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hois) != 0:
            for pred_hoi in pred_hois:
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0
                    for gt_hoi in gt_hois:
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids \
                           and pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'],
                           pred_hoi['category_id'])
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    vis_tag[gt_hois.index(max_gt_hoi)] =1
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                self.score[triplet].append(pred_hoi['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov=iou_mat.copy()
        iou_mat[iou_mat>=self.overlap_iou] = 1
        iou_mat[iou_mat<self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
            S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
                return intersect / (sum_area - intersect)
        else:
            return 0

    def triplet_nms_filter(self, preds):
        preds_filtered = []
        for img_preds in preds:
            pred_bboxes = img_preds['predictions']
            pred_hois = img_preds['hoi_prediction']
            all_triplets = {}
            for index, pred_hoi in enumerate(pred_hois):
                triplet = str(pred_bboxes[pred_hoi['subject_id']]['category_id']) + '_' + \
                          str(pred_bboxes[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

                if triplet not in all_triplets:
                    all_triplets[triplet] = {'subs':[], 'objs':[], 'scores':[], 'indexes':[]}
                all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
                all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
                all_triplets[triplet]['scores'].append(pred_hoi['score'])
                all_triplets[triplet]['indexes'].append(index)

            all_keep_inds = []
            for triplet, values in all_triplets.items():
                subs, objs, scores = values['subs'], values['objs'], values['scores']
                if self.use_soft_nms:
                    keep_inds = self.pairwise_soft_nms(np.array(subs), np.array(objs), np.array(scores))
                else:
                    keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

                if self.use_score_thres:
                    sorted_scores = np.array(scores)[keep_inds]
                    keep_inds = np.array(keep_inds)[sorted_scores > self.thres_score]

                keep_inds = list(np.array(values['indexes'])[keep_inds])
                all_keep_inds.extend(keep_inds)

            preds_filtered.append({
                'filename': img_preds['filename'],
                'predictions': pred_bboxes,
                'hoi_prediction': list(np.array(img_preds['hoi_prediction'])[all_keep_inds])
                })

        return preds_filtered

    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        order = scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            # mode 1, paper
            # ovr = (sub_inter + obj_inter) / (sub_union + obj_union)

            # mode 2
            #ovr = (sub_inter/sub_union) * (obj_inter / obj_union)
            ovr = np.power(sub_inter/sub_union, self.nms_alpha) * np.power(obj_inter / obj_union, self.nms_beta)
            inds = np.where(ovr <= self.thres_nms)[0]

            order = order[inds + 1]
        return keep_inds

    def pairwise_soft_nms(self, subs, objs, scores):
        assert subs.shape[0] == objs.shape[0]
        N = subs.shape[0]

        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        for i in range(N):
            tscore = scores[i]
            pos = i + 1
            if i != N - 1:
                maxpos = np.argmax(scores[pos:])
                maxscore = scores[pos:][maxpos]

                if tscore < maxscore:
                    subs[i], subs[maxpos.item() + i + 1] = subs[maxpos.item() + i + 1].copy(), subs[i].copy()
                    sub_areas[i], sub_areas[maxpos + i + 1] = sub_areas[maxpos + i + 1].copy(), sub_areas[i].copy()

                    objs[i], objs[maxpos.item() + i + 1] = objs[maxpos.item() + i + 1].copy(), objs[i].copy()
                    obj_areas[i], obj_areas[maxpos + i + 1] = obj_areas[maxpos + i + 1].copy(), obj_areas[i].copy()

                    scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].copy(), scores[i].copy()

            # IoU calculate
            sxx1 = np.maximum(subs[i, 0], subs[pos:, 0])
            syy1 = np.maximum(subs[i, 1], subs[pos:, 1])
            sxx2 = np.minimum(subs[i, 2], subs[pos:, 2])
            syy2 = np.minimum(subs[i, 3], subs[pos:, 3])
            
            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[pos:] - sub_inter
            sub_ovr = sub_inter/sub_union
    
            oxx1 = np.maximum(objs[i, 0], objs[pos:, 0])
            oyy1 = np.maximum(objs[i, 1], objs[pos:, 1])
            oxx2 = np.minimum(objs[i, 2], objs[pos:, 2])
            oyy2 = np.minimum(objs[i, 3], objs[pos:, 3])
            
            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[pos:] - obj_inter
            obj_ovr = obj_inter/obj_union

            # Gaussian decay
            ## mode 1
            #weight = np.exp(-(sub_ovr * obj_ovr) / self.soft_nms_sigma)

            ## mode 2
            weight = np.exp(-sub_ovr / self.soft_nms_sigma) * np.exp(-obj_ovr / self.soft_nms_sigma)

            scores[pos:] = weight * scores[pos:]

        # select the boxes and keep the corresponding indexes
        keep_inds = np.where(scores > self.soft_nms_thres_score)[0]

        return keep_inds


    def clip_preds_boxes(self, preds):
        preds_filtered = []
        for img_preds in preds:
            filename = img_preds['filename']

            input_file = os.path.join('data/hico_20160224_det/images/test2015/', filename)
            img = cv2.imread(input_file)
            h,w,c = img.shape

            pred_bboxes = img_preds['predictions']
            for pred_bbox in pred_bboxes:
                pred_bbox['bbox'] = self.bbox_clip(pred_bbox['bbox'], (h,w))

            preds_filtered.append(img_preds)

        return preds_filtered

    def bbox_clip(self, box, size):
        x1,y1,x2,y2 = box
        h,w = size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, w)
        y2 = min(y2, h)
        return [x1,y1,x2,y2]

