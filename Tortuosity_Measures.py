import math

import numpy as np


class TortuosityMeasures:


    def SCC(x_vec, y_vec):
        """
        Computes the Tortousity of a curve given by x_vec, y_vec with the Slope Chain Code method
        :param x_vec: numpy array X coordinates of the curve
        :param y_vec: numpy array Y coordinates of the curve
        :return the value of tortuosity given by the Slope Chain Code method
        """
        diff_x = np.diff(x_vec)
        diff_y = np.diff(y_vec)

        
        Theta = np.rad2deg(np.arctan2(diff_y, diff_x))
        alpha = np.diff(Theta)                

        trouble_idx = np.where(np.logical_or(alpha >= 180, alpha <= -180))
        trouble_values = alpha[trouble_idx]
        alpha[trouble_idx] = np.mod(trouble_values, np.sign(trouble_values) * (-360))
        alpha = alpha / 180

        SCC = np.sum(np.absolute(alpha))

        return [SCC, alpha]
    

    def SCC_Tree(scc_tree):
        acc = 0
        for k in range(2, len(scc_tree)):
            current = scc_tree[k]
            if current < 1:
                acc += current
                        
        tortuosity = acc / 2
        return tortuosity


    def SCC_Tree_1(interp_tree):
        """ Obtains the tortuosity of a tree representad by its (x,y) positions

        Args:
            interp_tree (_type_): a tree where each vertex is represented by its (x,y) positon

        Returns:
            _type_: the tortuosity value for all the tree
        """
        slope_change = [TortuosityMeasures.get_slope_diff(None, interp_tree[2], interp_tree[3])]
        last = interp_tree[2]
        current = interp_tree[3]

        for k in range(4, len(interp_tree) - 1):
            next = interp_tree[k]
            if type(next) is not tuple:
                if next == 1:
                    slope_change.append(1)
            else:
                alpha = TortuosityMeasures.get_slope_diff(last, current, next)
                slope_change.append(alpha)

                last = current
                current = next

        tortuosity = np.sum(np.absolute(slope_change)) / 2
        return tortuosity
    
 
    def get_slope_diff(p0, p1, p2):
        """
        """
        if p0 is None:
            Theta = np.rad2deg(np.arctan2(np.diff([p1[0], p2[0]]), np.diff([p1[1], p2[1]])))
            alpha = Theta[0]
        else:    
            dx = np.diff([p0[1], p1[1], p2[1]])
            dy = np.diff([p0[0], p1[0], p2[0]]) 
            Theta = np.rad2deg(np.arctan2(dy, dx))
            alpha = np.diff(Theta)[0]

            if np.logical_or(alpha >= 180, alpha <= -180):
                alpha = np.mod(alpha, np.sign(alpha) * (-360))
        
        return alpha / 180


    def DM_Tree(interp_tree):
        
        assert len(interp_tree) > 3, "The length of the input must be contain at least 4 elements."  
        
        tort = []
        p1 = interp_tree[2]
        p_i = p1 
        ini = 2
        end = 0        
        d_c = 0
        
        # traverse the whole tree             
        for k in range(3, len(interp_tree)):
            current = interp_tree[k]
            if type(current) is tuple:                     
                p_j = current
                d_c += math.dist(p_i, p_j)
                p_i = p_j 
                p_j = None
            else:
                if current == 1: continue
                end = current   
                p2 = p_i                
                if end > ini:                    
                    d_x = math.dist(p1, p2)
                    tort.append((d_c / d_x) - 1)        

                d_c = 0      
                p1 = p2          
                ini = end               

        mean_tort = np.average(tort)
        sum_tort = np.sum(tort)
        return [mean_tort, sum_tort, tort]