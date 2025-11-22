import math
import numpy as np

import SCC_Tree_old

from Tortuosity_Measures import TortuosityMeasures

class Morphology_Measures:

    @staticmethod
    def tree_branch_anlge(interp_tree):
        """
        """
        assert len(interp_tree) > 3, "The length of the input must be contain at least 4 elements."        
        coordinates = {}
        vertexes = []
        # find coordinates from bifurcations and ending points 
        for k in range(4, len(interp_tree) - 1):
            next = interp_tree[k]
            if type(next) is not tuple:     
                current = interp_tree[k - 1]
                if next == 1: continue
                if not coordinates.get(next):
                    coordinates[next] = current              
                vertexes.append(next)

        # find angles 
        angles = []        
        bifurcations = set()       
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
                
        median_angle = np.median(angles)
        return [median_angle, angles]
    

    def angle_between(b1, b2):
        v1 = (b1[1][0] - b1[0][0], b1[1][1] - b1[0][1])
        v2 = (b2[1][0] - b2[0][0], b2[1][1] - b2[0][1])
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        Theta = np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

        return Theta
    

    @staticmethod
    def tree_scc_branch_anlge(scc_tree):
        """
        Finds the bifurcation angle based only on the slope of the chain code of the given binary tree
        
        Returns:
            median_angle: double
                The mean of the all bifurcation angles found in the given tree
            anlges: list 
                A list with all the bifurcation angles found in the given tree
        """
        assert len(scc_tree) > 3, "The length of the input must be contain at least 4 elements."        

        sum_angle = 0
        scc_branch = {}
        vertexes = []  
        current_branch = []    
        p1 = None
        p2 = None

        # find coordinates from bifurcations and ending points 
        for k in range(2, len(scc_tree) - 1):
            next = scc_tree[k]            
            if next >= 1:     
                if next == 1: 
                    p1 = None
                    current_branch = []
                    sum_angle += 1
                    continue          
                vertexes.append(next)
                if not p1:
                    p1 = next
                else:
                    p2 = next

                if p1 and p2:
                    scc_branch[(p1,p2)] = current_branch
                    current_branch = []
                    p1 = p2
                    p2 = None
            else:
                sum_angle += next
                sum_angle %= 2
                
                if p1: 
                    current_branch.append(sum_angle)

        # find angles 
        angles = []        
        bifurcations = set()
        idx = 0
        for v in vertexes:
            if v in bifurcations:
                bifurcations.remove(v)
                b1 = (v, vertexes[idx - 1])
                b2 = (v, vertexes[idx + 1])
                v1 = np.average(np.array(scc_branch[b1]) * 180)
                v2 = np.average(np.array(scc_branch[b2]) * 180)
                
                if np.logical_or(v1 >= 180, v1 <= -180): v1 = np.mod(v1, np.sign(v1) * (-360))
                if np.logical_or(v2 >= 180, v2 <= -180): v2 = np.mod(v2, np.sign(v2) * (-360))
                
                theta = np.mod(v2 - v1, 180)  
                angles.append(theta)
            else:
                bifurcations.add(v)

            idx += 1
                
        mean_angle = np.mean(angles)
        return [mean_angle, angles]


    @staticmethod
    def tree_scc_branch_anlge_2(scc_tree):
        """
        Finds the bifurcation angle based only on the slope of the chain code of the any given tree
        
        Returns:
            median_angle: double
                The mean of the all bifurcation angles found in the given tree
            anlges: list 
                A list with all the bifurcation angles found in the given tree
        """
        assert len(scc_tree) > 3, "The length of the input must be contain at least 4 elements."        

        sum_angle = 0
        scc_branch = {}
        vertexes = []  
        current_branch = []    
        p1 = None
        p2 = None

        # find coordinates from bifurcations and ending points 
        for k in range(2, len(scc_tree) - 1):
            next = scc_tree[k]            
            if next >= 1:     
                if next == 1: 
                    p1 = None
                    current_branch = []
                    sum_angle += 1
                    continue          
                vertexes.append(next)
                if not p1:
                    p1 = next
                else:
                    p2 = next

                if p1 and p2:
                    scc_branch[(p1,p2)] = current_branch
                    current_branch = []
                    p1 = p2
                    p2 = None
            else:
                sum_angle += next
                sum_angle %= 2
                
                if p1: 
                    current_branch.append(sum_angle)

        bifurcations = []
        for (idx, x) in enumerate(scc_tree):
            if x > 1:
                if scc_tree[idx + 1] != 1:
                    bifurcations.append(x)
        bifurcations = bifurcations[0:-1]

        # find angles 

        branches = {}
        b = bifurcations.pop(0)
        for i in range(len(vertexes)):
            v = vertexes[i] 
            if v == b:
                next = vertexes[i + 1]
                if next > b:
                    branches.setdefault(b, []).append((b, next))
                if len(bifurcations) > 0:
                    b = bifurcations.pop(0)
                else:
                    break   

        angles = []
        for key in branches:
            v_branches = branches[key]
            for i in range(len(v_branches) - 1):
                b1 = v_branches[i]
                b2 = v_branches[i + 1]
                v1 = np.average(np.array(scc_branch[b1]) * 180)
                v2 = np.average(np.array(scc_branch[b2]) * 180)
                
                if np.logical_or(v1 >= 180, v1 <= -180): v1 = np.mod(v1, np.sign(v1) * (-360))
                if np.logical_or(v2 >= 180, v2 <= -180): v2 = np.mod(v2, np.sign(v2) * (-360))
                
                theta = np.mod(v2 - v1, 180)  
                angles.append(theta)
                
        mean_angle = np.mean(angles)
        return [mean_angle, angles]


    @staticmethod
    def tree_length(interp_tree):

        assert len(interp_tree) > 3, "The length of the input must be contain at least 4 elements."                
        length = 0
        p1 = interp_tree[2]
        # traverse the whole tree     
        for k in range(3, len(interp_tree) - 1):
            next = interp_tree[k]
            if type(next) is tuple:     
                if next != 1: 
                    p2 = next
                    length += math.dist(p1,p2)
                    p1 = p2 
                    p2 = None

        length /= 2
        return length    
    

    @staticmethod
    def tree_branch_length(interp_tree):

        assert len(interp_tree) > 3, "The length of the input must be contain at least 4 elements."  
        length = 0
        lengths = []
        p1 = interp_tree[2]
        ini = 2
        end = 0
        # traverse the whole tree     
        for k in range(3, len(interp_tree) - 1):            
            current = interp_tree[k]
            if type(current) is tuple:                     
                p2 = current
                length += math.dist(p1,p2)
                p1 = p2 
                p2 = None
            else:
                if current == 1: continue
                end = current
                if end > ini:
                    lengths.append(length)    
                length = 0
                ini = end               

        mean_length = np.average(lengths)
        sum_length = np.sum(lengths)
        return [mean_length, sum_length, lengths]
    

    @staticmethod
    def tree_scc_count_features(scc_tree):
        assert len(scc_tree) > 3, "The length of the input must be contain at least 4 elements." 

        vertexes = [] 
        # find coordinates from bifurcations and ending points 
        for k in range(2, len(scc_tree) - 1):
            current = scc_tree[k]
            next = scc_tree[k + 1]
            if current >= 1:                     
                vertexes.append(current)
 
        terminals = 1
        for v in vertexes:            
            if v == 1: 
                terminals += 1

        verts = np.max(vertexes) - 1

        segments = verts - 1
        bifurcations = verts - terminals

        return [segments, bifurcations, terminals]
    

    @staticmethod
    def tree_scc_circularity(scc_tree):
        A_m = Morphology_Measures.slope_change_mean(scc_tree)

        # obtain the circularity        
        T_c = 0
        for k in range(1, len(scc_tree)):            
            a_i = scc_tree[k]
            if a_i <= 1:     
               T_c += abs(a_i - A_m)      
            # else:
            #     print("Fuck you {}".format(a_i))

        norm_circ = T_c / len(scc_tree)
        return [T_c, norm_circ]

    
    @staticmethod
    def tree_scc_circularity_2(scc_tree):
        A_m = Morphology_Measures.slope_change_mean_2(scc_tree)

        tree = scc_tree.tree
        # obtain the circularity        
        T_c = 0
        for k in range(1, len(tree)):
            a_i = tree[k]
            if a_i <= 1:     
               T_c += abs(a_i - A_m)

        norm_circ = T_c / len(scc_tree)
        return [T_c, norm_circ]


    @staticmethod
    def curve_scc_circularity(scc_curve):
        acc = 0
        n = 0
        for slope in scc_curve:            
            acc += slope
            n += 1        

        A_m = acc / n        

        # obtain the circularity        
        A_c = 0
        for slope in scc_curve:  
            A_c += abs(slope - A_m)

        return A_c
    

    @staticmethod
    def tree_scc_linearity(scc_tree):        
        [seg, bifur, term] = Morphology_Measures.tree_scc_count_features(scc_tree)

        T_l = 0
        b = {}
        for k in range(1, len(scc_tree)):
            current = scc_tree[k]
            if current < 1:     
                T_l += abs(current)
            elif current > 1:
                if b.get(current):
                    b[current].append(scc_tree[k + 1])
                else:
                    b[current] = [scc_tree[k + 1]]

        acc_bif = 0
        for key in b:
            acc_bif += -sum(b[key])

        return (T_l / 2) - acc_bif
    

    @staticmethod
    def slope_change_mean(scc_tree):
        n = len(scc_tree)
        SC_m = 2 / n
        return SC_m
    

    @staticmethod
    def slope_change_mean_2(scc_tree):
        n = scc_tree.size
        SC_m = 2 / n
        return SC_m


    @staticmethod
    def convex_concav(scc_tree):
        acc = 0
        slope_acc = 0
        
        for k in range(2, len(scc_tree)):
            current = scc_tree[k]
            if current < 1:
                slope_acc += abs(current)
                acc += current
                        
        t = slope_acc / 2
        acc /= 2
        C_m = t - abs(acc)
        C = None      
          
        if abs(acc) == t:
            C = 1
        elif abs(acc) < t:
            C = -1
        else:
            C = 0

        
        return [C, C_m]


    