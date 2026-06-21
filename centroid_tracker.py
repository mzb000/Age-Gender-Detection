import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=40):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.colors = OrderedDict()
        self.max_disappeared = max_disappeared
        self._color_pool = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
            (0, 128, 255), (128, 0, 255), (255, 0, 128), (0, 255, 128),
        ]

    def register(self, centroid, bbox):
        self.objects[self.next_id] = (centroid, bbox)
        self.disappeared[self.next_id] = 0
        self.colors[self.next_id] = self._color_pool[self.next_id % len(self._color_pool)]
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.colors[object_id]

    def update(self, bboxes):
        if len(bboxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.colors

        input_centroids = np.zeros((len(bboxes), 2), dtype=int)
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            input_centroids[i] = ((x1 + x2) // 2, (y1 + y2) // 2)

        if len(self.objects) == 0:
            for i in range(len(bboxes)):
                self.register(input_centroids[i], bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid][0] for oid in object_ids])

            D = dist.cdist(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = (input_centroids[col], bboxes[col])
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], bboxes[col])

        return self.objects, self.colors
