import numpy as np

def _precision_recall_f1(right, predict, total):
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f

def evaluate(epoch_predicts_domains, epoch_labels_domains, epoch_predicts_areas, epoch_labels_areas, domain_vocab, area_vocab, threshold=0.5, top_k=None):
    assert len(epoch_predicts_domains) == len(epoch_labels_domains), 'Domain prediction and label size mismatch.'
    assert len(epoch_predicts_areas) == len(epoch_labels_areas), 'Area prediction and label size mismatch.'

    domain_label2id = domain_vocab['v2i']
    domain_id2label = domain_vocab['i2v']
    area_label2id = area_vocab['v2i']
    area_id2label = area_vocab['i2v']

    right_total, predict_total, gold_total = 0, 0, 0
    area_right_count_list = [0 for _ in range(len(area_label2id.keys()))]
    area_gold_count_list = [0 for _ in range(len(area_label2id.keys()))]
    area_predicted_count_list = [0 for _ in range(len(area_label2id.keys()))]

    for domain_pred, domain_true, area_pred, area_true in zip(epoch_predicts_domains, epoch_labels_domains, epoch_predicts_areas, epoch_labels_areas):
        np_domain_pred = np.array(domain_pred, dtype=np.float32)
        domain_predict_descent_idx = np.argsort(-np_domain_pred)

        # domain 예측이 맞았는지 확인
        correct_domain = (domain_predict_descent_idx[0] == domain_true[0])

        if correct_domain:  # domain 예측이 맞았을 때만 area 평가
            np_area_pred = np.array(area_pred, dtype=np.float32)
            area_predict_descent_idx = np.argsort(-np_area_pred)
            area_predict_id_list = []
            if top_k is None:
                top_k = len(area_pred)
            for j in range(top_k):
                if np_area_pred[area_predict_descent_idx[j]] > threshold:
                    area_predict_id_list.append(area_predict_descent_idx[j])

            for gold in area_true:
                area_gold_count_list[gold] += 1
                for pred in area_predict_id_list:
                    if gold == pred:
                        area_right_count_list[gold] += 1

            for pred in area_predict_id_list:
                area_predicted_count_list[pred] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()

    for i, label in area_id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(area_right_count_list[i],
                                                                                             area_predicted_count_list[i],
                                                                                             area_gold_count_list[i])
        right_total += area_right_count_list[i]
        gold_total += area_gold_count_list[i]
        predict_total += area_predicted_count_list[i]

    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))

    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total if gold_total > 0 else 0.0
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'micro_f1': micro_f1, 'macro_f1': macro_f1}