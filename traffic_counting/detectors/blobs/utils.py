def get_centroid(bbox):
    
    return (round((bbox[0] + bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[1] + bbox[3]) / 2))

def box_contains_point(bbox, pt):
    return bbox[0] < pt[0] < bbox[0] + bbox[2] and bbox[1] < pt[1] < bbox[1] + bbox[3]

def get_area(bbox):
   
    return bbox[2] * bbox[3]
def get_classID(bbox):
    return bbox[4]