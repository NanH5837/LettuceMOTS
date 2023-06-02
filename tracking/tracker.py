import numpy as np
from filterpy.kalman import KalmanFilter
from feature_extraction import feature_extraction
from matching import matching
import global_config as config

np.random.seed(0)

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01   #0.01
    self.kf.Q[4:,4:] *= 0.01   #0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.feature = []
    self.mask = []
    self.state = 0
    self.kfpredict = []

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.history = []
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

class Tracker(object):
  count = 0

  def __init__(self):

    self.tracks = []

  def update(self, masks, dets, storage):
    detections, masks = feature_extraction(masks, dets)
    trks = np.zeros((len(storage), len(detections[0])))

    for t, trkk in enumerate(trks):
      if storage[t].feature[0] > 0:
        pos = storage[t].predict()[0]
        box = [pos[0], pos[1], pos[2], pos[3]]
        storage[t].kfpredict[:4] = box

    matches, unmatched_detections, unmatched_tracks = matching(detections, storage)

    for m in matches:

      storage[m[0]].feature = detections[m[1]]
      storage[m[0]].mask = masks[m[1]]
      storage[m[0]].update(detections[m[1], :])
      storage[m[0]].state = 1
      self.tracks.append(storage[m[0]])

    for ud in unmatched_detections:
      trk = KalmanBoxTracker(detections[ud, :])
      trk.mask = masks[ud]
      trk.feature = detections[ud, :]
      trk.state = 1
      self.tracks.append(trk)
      storage.append(trk)

    for ut in unmatched_tracks:
      storage[ut].state = 0

    config.min = int(self.tracks[0].id)
    config.max = int(self.tracks[-1].id)

    if (len(self.tracks) > 0):
      return self.tracks, storage
