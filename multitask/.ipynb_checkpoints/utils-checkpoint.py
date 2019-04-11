import json


def loadjson(fname):
  with open(fname) as j:
    f = json.load(j)
  return f
