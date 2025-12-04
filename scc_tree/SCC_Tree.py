import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import utils.Smoothing as smooth

matplotlib.use('Qt5Agg')

class SCC_Tree:
    """A class to build and represent a tree by its back and forht chain"""
    
    size = 0
    size_without_terminals = 0

    _tree = []
    _dists = []
    _scc_chain_tree = []
    _scc_chain_dist = [] 

    _raster_tree = []


    def __init__(self, interp_tree):
        [self._tree, self._dists] = self._build_scc_tree(interp_tree)
        self._create_scc_chain()
        self.size = self._tree.__len__()
        # self.size_without_terminals = self.__get_size_without_terminals()
        # 



    @classmethod
    def create_from_image(cls, image, tree_root) -> 'SCC_Tree':
        cls._treepath = cls._build_tree(image, tree_root)
        interp_tree = cls._build_interpolated_tree(cls._treepath)
        scc_tree = cls(interp_tree)   
        return scc_tree
        # return cls(interp_tree)   



    @staticmethod
    def _build_tree(image, root):
        """ 
        Form an image containing a skeletonized tree, return a position (x,y) list result of a hamiltonian traverse of the tree with bifurcations and 
        terminal vertexes marked.

        Args:
            _image (numpy.array): representing a BW image with a skeletonized tree in white pixels (255)
            root (_type_): the (y,x) position of a terminal vertex where the treaverse begins 

        Returns:
            _type_: _description_
        """

        cp = root  # moving Position
        p_idx = 0
        p_vec = ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))

        tree = [2, 1, root]
        sp_idx = 2

        bifurcations = {}

        while True:
            for i in range(8):
                k = (p_idx + i) % 8
                p = p_vec[k]
                tp = (cp[0] + p[0], cp[1] + p[1])

                if tp[0] < len(image) and tp[1] < len(image[0]):
                    if image[tp[0]][tp[1]] != 0:
                        if k % 2 != 0:
                            kk = (k + 1) % 8
                            if image[cp[0] + p_vec[kk][0]][cp[1] + p_vec[kk][1]] != 0:
                                tp = (cp[0] + p_vec[kk][0], cp[1] + p_vec[kk][1])

                        p_idx = k - 2
                        cp = tp
                        tree.append(cp)

                        is_special = SCC_Tree._is_special_point(image, cp)
                        if is_special[0]:
                            if is_special[1] > 2:
                                if SCC_Tree._is_line_junction(image, cp):
                                    if not bifurcations.get(cp):
                                        sp_idx += 1
                                        bifurcations[cp] = sp_idx
                                        tree.append(bifurcations[cp])
                                    else:
                                        tree.append(bifurcations[cp])

                            if is_special[1] == 1 and cp != root:
                                sp_idx += 1
                                tree.append(sp_idx)
                                tree.append(1)
                        break
            if cp == root:
                tree.append(2)
                break

        return tree
    

    
    @staticmethod
    def _is_special_point(image, p):
        """
        Test if the given point is a bifurcation or end point

        :param image: binary image to analyze
        :param p:  current point
        :return: a tuple that means fist position 0, 1 if is or not a special point, second position the number of
        connections of the point
        """
        count = 0
        neighborhood = ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))
        for i in range(8):
            shift = neighborhood[i]
            tp = np.array((p[0] + shift[0], p[1] + shift[1]))
            if image[tp[0]][tp[1]] != 0:
                count += 1

        if count > 2:
            return 1, count
        elif count < 2:
            return 1, count
        else:
            return 0, count
        


    @staticmethod
    def _is_line_junction(image, p):
        # define junction patterns
        pattterns = (
            (0, 1, 0, 0, 1, 0, 0, 1),
            (1, 0, 1, 0, 0, 1, 0, 0),
            (0, 1, 0, 1, 0, 0, 1, 0),
            (0, 0, 1, 0, 1, 0, 0, 1),
            (1, 0, 0, 1, 0, 1, 0, 0),
            (0, 1, 0, 0, 1, 0, 1, 0),
            (0, 0, 1, 0, 0, 1, 0, 1),
            (1, 0, 0, 1, 0, 0, 1, 0),
            (0, 0, 0, 1, 0, 1, 0, 1),
            (0, 1, 0, 0, 0, 1, 0, 1),
            (0, 1, 0, 1, 0, 0, 0, 1),
            (0, 1, 0, 1, 0, 1, 0, 0),
            (0, 1, 0, 1, 0, 1, 0, 1),
            (1, 0, 1, 0, 1, 0, 1, 0),
            (0, 0, 1, 0, 1, 0, 1, 0),
            (1, 0, 0, 0, 1, 0, 1, 0),
            (1, 0, 1, 0, 0, 0, 1, 0),
            (1, 0, 1, 0, 1, 0, 0, 0)
        )

        for pat in pattterns:
            neighborhood = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
            for i in range(8):
                if pat[i] == 1:
                    shift = neighborhood[i]
                    tp = (p[0] + shift[0], p[1] + shift[1])
                    if image[tp[0]][tp[1]] == 0:
                        break
                if pat[i] == 0:
                    shift = neighborhood[i]
                    tp = (p[0] + shift[0], p[1] + shift[1])
                    if image[tp[0]][tp[1]] == 255:
                        break
            if i == 7:
                return True

        return False
    


    @staticmethod
    def _build_interpolated_tree(tree_path) -> list:
        """

        :param tree_path:
        :return:
        """
        p1 = 2
        p2 = 0
        branch = []
        branches = {}
        interp_branches = {}
        interp_tree = [2, 1, tree_path[2]]

        for k in range(2, len(tree_path)):
            data = tree_path[k]

            if type(data) is not tuple:
                # print(branch[-1])
                if data != 1:
                    if not p1:
                        p1 = data
                    else:
                        p2 = data
                if p1 and p2:
                    if p1 < p2:       
                        smooth_curve = smooth.smooth_with_regularization(np.array(branch), arclen_points=0.59, smoothing_factor=0.057)
                        interp_branch = [(point[0], point[1]) for point in smooth_curve]

                        branches[(p1, p2)] = branch
                        interp_branches[(p1, p2)] = interp_branch

                        for i in range(1, len(interp_branch)):
                            interp_tree.append(interp_branch[i])
                    else:
                        interp_branch = interp_branches[(p2, p1)]
                        interp_branch.reverse()
                        for i in range(1, len(interp_branch)):
                            interp_tree.append(interp_branch[i])

                    branch = [branch[-1]]
                    p1 = p2
                    p2 = 0
                interp_tree.append(data)
            else:
                branch.append(data)

        return interp_tree


    @staticmethod
    def _build_scc_tree(interp_tree) -> list[list, list]:
        tree_dist = [2, 1, math.dist(interp_tree[2], interp_tree[3])]
        tree_scc = [2, 1, SCC_Tree._get_slope_change(None, interp_tree[2], interp_tree[3])]
        last = interp_tree[2]
        current = interp_tree[3]

        for k in range(4, len(interp_tree) - 1):
            next = interp_tree[k]
            if type(next) is not tuple:
                if next != 1:
                    tree_scc.append(next)
                    tree_dist.append(next)            
            else:
                if last == next: 
                    alpha = 1
                else:
                    alpha = SCC_Tree._get_slope_change(last, current, next)
                dist = math.dist(current, next)
                tree_scc.append(alpha)
                tree_dist.append(dist)

                last = current
                current = next

        tree_scc.append(-tree_scc[2])
        tree_dist.append(tree_dist[2])
    
        return [tree_scc, tree_dist]

    
    @staticmethod
    def _get_slope_change(p0, p1, p2):
        """
        :param p0: previous slope change
        :param p1: first point (from)
        :param p2: second point (to)
        """
        if p0 is None:
            alpha = np.rad2deg(np.arctan2(p2[0] - p1[0], p2[1] - p1[1]))
        else:    
            dx = np.diff([p0[1], p1[1], p2[1]])
            dy = np.diff([p0[0], p1[0], p2[0]]) 
            Theta = np.rad2deg(np.arctan2(dy, dx))
            alpha = Theta[0] - Theta[1]

            if np.logical_or(alpha >= 180, alpha <= -180):
                alpha = np.mod(alpha, np.sign(alpha) * (-360))
        
        alpha_n = alpha / 180
        if abs(alpha_n) < 1e-14:
            alpha_n = 0

        return alpha_n 
    


    def _create_scc_chain(self):
        self._scc_chain_tree = []
        self._scc_chain_dist = []

        for idx, val in enumerate(self._tree):
            if val <= 1:
                self._scc_chain_tree.append(self._tree[idx])
                self._scc_chain_dist.append(self._dists[idx])
        


    def slope_change_mean(self):
        n = self.size
        SC_m = 2 / n
        return SC_m
        


    def tree_tortuosity(self):
        """
        Calculate the tortuosity of the tree represented by the SCC chain
        Returns: [tortuosity, tort_norm] the tortuosity and the normalized tortuosity
        """
        slope_acc = 0
        count = 0

        for v in self._scc_chain_tree:            
            if v < 1:
                slope_acc += abs(v)
                count += 1
                        
        tortuosity = slope_acc / 2
        tort_norm = tortuosity / count

        return [tortuosity, tort_norm]
    


    def tree_length(self):

        assert len(self._scc_chain_tree) > 3, "The length of the input must be contain at least 4 elements."                
        length = 0
        for slope, dist in zip(self._scc_chain_tree, self._scc_chain_dist):      
            length += dist
        
        return length / 2
    


    def tree_non_circularity(self):
        """
        Calculate the non-circularity of the tree represented by the SCC chain
        Returns: [D_c, Dc_norm] the non-circularity and the normalized non-circularity
        """
        A_m = self.slope_change_mean()

        D_c = 0
        for c_i in self._scc_chain_tree:
            D_c += abs(A_m - c_i)

        Dc_norm = D_c / len(self._scc_chain_tree)
        return [D_c, Dc_norm]
    


    def __get_size_without_terminals(self) -> int:
        """Get the amount of chain elements whose numerical values are less than one, 
            i.e. without the terminal nodes of the tree
        Returns:
            int: the amount of chain elements whose numerical values are less than one
        """

        if self.tree:
            n = 0        
            for k in range(1, len(self.tree)):
                current = self.tree[k]
                if current < 1:                                
                    n += 1                    
            return n
        

    def __get_size(self) -> int:
        """ Get the amount of chain elements in the tree representation

        Returns:
            int: the amount of chain elements
        """
        if self.tree:
            n = 0        
            for k in range(1, len(self.tree)):
                current = self.tree[k]
                if current <= 1:                                
                    n += 1                    
            return n
        

    @staticmethod
    def prune_tree(tree):
        """ From the original string construct the extended string that contains 
        the numbered bufurcations (NOT FINISHED YET)

        Args:
            tree (list): the list containing the original tree descriptor  

        Returns:
            list: a list containting the extended tree descriptor 
        """
        # find leafs and it's branches        
        bifurcations = {}            
        
        pruned_tree = list(tree)
        pruned_idxs = range(0, len(tree))

        b_idx = 3
        while pruned_tree.count(1) > 2:       
            idx = 1
            pivot = -1            
            prune = {}
            temp_bifur = {}
            nodes_path = []
            visited = []
            while idx < len(pruned_tree):            
                nodes_path.append(idx)
                e = pruned_tree[idx]
                if e == 1:
                    pivot = idx
                    inner_idx = 1
                    nodes_path.pop()
                    while inner_idx > 0:
                        p1 = pivot - inner_idx
                        p2 = pivot + inner_idx
                        e1 = pruned_tree[p1]
                        e2 = pruned_tree[p2]
                        # if e1 == -e2:
                        if math.isclose(e1, -e2, rel_tol=1e-12):
                            inner_idx += 1
                            nodes_path.pop()
                        else:
                            visited.append([len(nodes_path), (p1,p2)])
                            bifurcations[pruned_idxs[p1]] = pruned_idxs[p2]       
                            temp_bifur[p1] = p2                                 
                            node = prune.get(p1)
                            if node:                     
                                prune.pop(p1)
                                prune[p2] = node
                            else:                            
                                prune[p2] = p1                                                                                                    
                            inner_idx = 0
                            idx = p2
                idx += 1
            
            # mark bifurcations as a new leafs
            skip = []

            # while visited:           
            #     (k,v) = visited.popitem()
            #     slope_sum = pruned_tree[k] + pruned_tree[v]
            #     while visited.get(v):
            #          v = visited.pop(v)
            #          slope_sum += pruned_tree[v]
            #     if not(math.isclose(slope_sum, round(slope_sum), rel_tol=1e-12)):
            #         skip.append(e)

            i1 = 0
            i2 = 0
            nodes = []
            idxs = []
            for p2, p1 in prune.items():            
                i2 = p1
                pruned_tree[p1] = 1       
                nodes.extend(pruned_tree[i1:i2+1])
                idxs.extend(pruned_idxs[i1:i2+1])                         
                i1 = p2 + 1

            nodes.extend(pruned_tree[i1:len(pruned_tree)])
            idxs.extend(pruned_idxs[i1:len(pruned_tree)])  

            pruned_tree = list(nodes)
            pruned_idxs = list(idxs)

        return bifurcations



    @staticmethod
    def create_extended_tree(tree):
        extend_string = list(tree)        
        bifurcations = SCC_Tree.prune_tree(tree)
        sorted_bifurcations = sorted(bifurcations.items())

        idx = 1
        ext_idx = 0
        b_idx = 3
        node_bifur_idx = {}
        while idx < len(tree):            
            e = tree[idx]
            if bifurcations.get(idx):
                extend_string.insert(idx + ext_idx, b_idx)
                node_bifur_idx[idx] = b_idx
                b_idx += 1
                ext_idx += 1

            if e == 1:
                pivot = idx
                inner_idx = 1
                extend_string.insert(idx + ext_idx, b_idx)
                b_idx += 1
                ext_idx += 1
                while inner_idx > 0:
                    p1 = pivot - inner_idx
                    p2 = pivot + inner_idx
                    e1 = tree[p1]
                    e2 = tree[p2]
                    if math.isclose(e1, -e2):
                        inner_idx += 1
                    else:
                        bi = node_bifur_idx[p1]
                        node_bifur_idx[p2] = bi
                        extend_string.insert(p2 + ext_idx, bi)
                        ext_idx += 1
                        inner_idx = 0
                        idx = p2
            idx += 1

        return extend_string
    

    @staticmethod
    def create_extended_tree_2(tree):
        pruned_tree = list(tree)
        pruned_idxs = range(0, len(tree))
        bifurcations = set()
        while len(pruned_tree) > 1:       
            idx = 1
            hold = {}
            finished = False
            finished_t = False
            while not finished:    
                e = pruned_tree[idx]
                if e == 1:
                    pivot = idx
                    inner_idx = 1
                    while inner_idx > 0:
                        p1 = pivot - inner_idx
                        p2 = pivot + inner_idx 
                        r1 = pruned_idxs[pivot - inner_idx]
                        r2 = pruned_idxs[pivot + inner_idx] 
                        e1 = pruned_tree[p1]
                        e2 = pruned_tree[p2]
                        finished_t = True if not finished_t and idx == 0 else finished_t                          
                        if math.isclose(e1, -e2, rel_tol=1e-12):
                            inner_idx += 1                            
                        else:                                    
                            if not hold or hold.get(r1):
                                if finished_t and hold.get(r2):
                                    finished = True
                                    break
                                hold[r2] = r1                                 
                            else:
                                if finished_t:
                                    finished = True
                                    break
                                hold.clear()
                                hold[r2] = r1                                                             
                            inner_idx = 0
                            idx = p2
                            
                idx = (idx + 1) % len(pruned_tree)
                
            ## prune branches
            nodes = list(pruned_tree)
            indexes = list(pruned_idxs)
            for r2, r1 in hold.items():     
                bifurcations.add(r1)
                bifurcations.add(r2)
                p1 = pruned_idxs.index(r1)
                p2 = pruned_idxs.index(r2)            
                for idx in range(p1, p1 + ((p2 - p1) % len(nodes))):
                    idx = idx % len(nodes)
                    nodes[idx] = -1                            
            nodes[p2] = 1     

            pruned_tree = []
            pruned_idxs = []
            for i, e in zip(indexes, nodes):
            
                if e != -1:
                    pruned_tree.append(e)
                    pruned_idxs.append(i)
        
        return sorted(list(bifurcations))
    


    @staticmethod
    def create_extended_tree_3(tree):
        """
        Create an extended tree representation without grade restrictions.
        Args:
            tree (list): the list containing the original tree descriptor
        Returns:
            list: the extended tree representation
        """
        
        bifurcations = {}
        bifurcations_counter = set()      
        slope_acc = 0
        backward = False
        visited_idx = []
        slope_acc_position = [None] * len(tree)
        slope_acc_position[0] = 0.0

        for idx in range(1, len(tree)):
            slope = tree[idx]  
            slope_acc += slope
            slope_acc_position[idx] = slope_acc

            if slope_acc > 1.0: 
                slope_acc_position[-1] = slope_acc % -1.0
            if slope_acc < -1.0: 
                slope_acc_position[-1] = slope_acc % 1.0
            
            if slope == 1.0:
                backward = True
            else:
                if backward:
                    last_visited_idx = visited_idx.pop()
                    last_visited_slope = tree[last_visited_idx] 
                    if math.isclose(slope, -last_visited_slope, rel_tol=1e-12): 
                        pass
                    else:                                                                      
                        bifurcations_counter.add(idx)
                        bifurcations_counter.add(last_visited_idx)
                        
                        bifurcations[idx] = last_visited_idx
                        if last_visited_idx in bifurcations:
                            pass                            

                        a = abs(slope_acc_position[visited_idx[-1] - 1] - slope_acc)
                        if math.isclose(a, 1.0, rel_tol=1e-12):
                            pass                            
                        else:
                            backward = False
                            visited_idx.append(idx)                                                    
                else:
                    visited_idx.append(idx)
                    
        return sorted(list(bifurcations_counter))        


    def get_tree(self):
        return self._tree
    
    
    def plot_tree(self):

        yy1 = 0
        xx1 = -8
        slope = 0        
        segment_size = 1
        points = []
        X = []
        Y = []
        k = 2
        while k < len(self._tree):
            val = self._tree[k]
            segment_size = self._dists[k]

            if val < 2:
                X.append(xx1)
                Y.append(yy1)

                s = np.double(val)
                slope = slope + s
                xx1 = xx1 + segment_size * np.cos(slope * np.pi)
                yy1 = yy1 + segment_size * np.sin(slope * np.pi)
            k += 1

        plt.figure(figsize=(12, 12))
        plt.plot(X, Y, 'r.-')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        return [X, Y]
    
    
    def get_pixelated_branches(self):
        p1 = 2
        p2 = 0
        branch = []
        branches = []
        
        for k in range(2, len(self._treepath)):
            data = self._treepath[k]

            if type(data) is not tuple:
                if data != 1:
                    if not p1:
                        p1 = data
                    else:
                        p2 = data
                if p1 and p2:
                    if p1 < p2:       
                        branches.append(np.array(branch))                    

                    branch = [branch[-1]]
                    p1 = p2
                    p2 = 0                
            else:
                branch.append(data)

        return branches
