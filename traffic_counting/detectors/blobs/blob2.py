from blobs.utils import get_centroid, get_area, get_classID


class Blob:
    def __init__(self, _bounding_box, _tracker):
        self.bounding_box = _bounding_box
        self.centroid = get_centroid(_bounding_box)
        self.area = get_area(_bounding_box)
        self.tracker = _tracker
        self.num_consecutive_tracking_failures = 0
        self.counted = False
        self.classID = get_classID(_bounding_box)

    def update(self, _bounding_box, _tracker=None):
        self.bounding_box = _bounding_box
        self.centroid = get_centroid(_bounding_box)
        self.area = get_area(_bounding_box)
        self.classID = get_classID(_bounding_box)
        if _tracker:
            self.tracker = _tracker