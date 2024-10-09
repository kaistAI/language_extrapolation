from .base import ObjectPatch
from matplotlib import patches

class RectanglePatch(ObjectPatch):

    def __init__(self, loc, **kwargs):
        super().__init__(patches.Rectangle, loc, 1, 1, **kwargs)

        
    def _correction(self, loc):
        return (loc[0], loc[1])
    
class CirclePatch(ObjectPatch):

    def __init__(self, loc, r, **kwargs):
        super().__init__(patches.Circle, loc, r **kwargs)        
        self.dx = self.dy = r