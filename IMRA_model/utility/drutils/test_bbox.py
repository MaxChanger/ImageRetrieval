from projects.drutils.bbox import calculate_union_area, calculate_union_area_v2
import numpy as np


def test_area():
    np.random.seed(12)
    max_pos = 10 ** 4
    num_box = 30
    boxes = np.random.randint(max_pos, size=(num_box, 4))
    for box in boxes:
        box[0], box[2] = min(box[0], box[2]), max(box[0], box[2])
        box[1], box[3] = min(box[1], box[3]), max(box[1], box[3])

    for i in range(num_box):
        partial = boxes[:i + 1, ]
        print(i)
        result_1 = calculate_union_area(partial)
        result_2 = calculate_union_area_v2(partial)
        print(result_1, result_2)
        assert (int(result_1) == int(result_2))


if __name__ == '__main__':
    test_area()
