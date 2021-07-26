import time

class PID:
  def __init__(self, p, i, d):
    self.kP, self.kI, self.kD = p, i, d
    self.time_curr = time.time()
    self.time_prev = self.time_curr
    self.error_prev = 0

    self.cP, self.cI, self.cD = 0, 0, 0

  def update (self, error):  
    self.time_curr = time.time()
    time_d = self.time_curr - self.time_prev
    error_d = error - self.error_prev

    self.cP = error
    self.cI += error * time_d
    if (time_d > 0):
      self.cD = error_d/time_d
    else:
      self.cD = 0

    self.time_prev = self.time_curr
    self.error_prev = error

    u = self.kP * self.cP + self.kI * self.cI + self.kD * self.cD
    #print(u, self.cP, self.cI, self.cD)
    return u
