import math
import numpy as np

from Tortuosity_Measures import TortuosityMeasures

class Morphology_Measures:

    @staticmethod
    def three_branch_anlge(scc_tree):
        """
        """
        assert len(scc_tree) > 3, "The length of the input must be contain at least 4 elements."        
        coordinates = {}
        vertexes = []
        # find coordinates from bifurcations and ending points 
        for k in range(4, len(scc_tree) - 1):
            next = scc_tree[k]
            if type(next) is not tuple:     
                current = scc_tree[k - 1]
                if next == 1: continue
                if not coordinates.get(next):
                    coordinates[next] = current              
                vertexes.append(next)

        # find angles 
        angles = []        
        bifurcations = set()
        b1 = tuple()
        b2 = tuple()
        idx = 0
        for v in vertexes:
            if v in bifurcations:
                bifurcations.remove(v)
                b1 = (coordinates[v], coordinates[vertexes[idx - 1]])
                b2 = (coordinates[v], coordinates[vertexes[idx + 1]])                    
                Theta = Morphology_Measures.angle_between(b1, b2)                        
                angles.append(Theta)
            else:
                bifurcations.add(v)

            idx += 1
                
        mean_angle = np.mean(angles[0:-1])
        return mean_angle
    

    def angle_between(b1, b2):
        v1 = (b1[1][0] - b1[0][0], b1[1][1] - b1[0][1])
        v2 = (b2[1][0] - b2[0][0], b2[1][1] - b2[0][1])
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        Theta = np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

        return Theta