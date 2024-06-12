import math
import numpy as np

class SCC_Tree:
    
    size = 0
    tree = []
    dists = []

    def __init__(self, tree_path) -> None:
        [self.tree, self.dists] = self.__build_scc_tree(tree_path)
        self.size = self.__get_size()


    def __build_interpolated_tree(self, tree_path) -> list:
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
                if data != 1:
                    if not p1:
                        p1 = data
                    else:
                        p2 = data
                if p1 and p2:
                    if p1 < p2:
                        bx = np.array([point[1] for point in branch])
                        by = np.array([point[0] for point in branch])
                        D = self.__interp_curve(round(len(bx) * 0.25), by, bx)
                        interp_branch = [(point[0], point[1]) for point in D]

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
    

    def __interp_curve(self, n, px, py):
        if n > 1:
            # equally spaced in arclength
            N = np.transpose(np.linspace(0, 1, n))

            # how many points will be uniformly interpolated?
            nt = N.size

            # number of points on the curve
            n = px.size
            pxy = np.array((px, py)).T
            p1 = pxy[0, :]
            pend = pxy[-1, :]
            last_segment = np.linalg.norm(np.subtract(p1, pend))
            epsilon = 10 * np.finfo(float).eps

            # IF the two end points are not close enough lets close the curve
            # if last_segment > epsilon * np.linalg.norm(np.amax(abs(pxy), axis=0)):
            #     pxy = np.vstack((pxy, p1))
            #     nt = nt + 1
            # else:
            #     print('Contour already closed')

            pt = np.zeros((nt, 2))

            # Compute the chordal arclength of each segment.
            chordlen = (np.sum(np.diff(pxy, axis=0) ** 2, axis=1)) ** (1 / 2)
            # Normalize the arclengths to a unit total
            chordlen = chordlen / np.sum(chordlen)
            # cumulative arclength
            cumarc = np.append(0, np.cumsum(chordlen))

            tbins = np.digitize(N, cumarc)  # bin index in which each N is in

            # catch any problems at the ends
            # tbins[np.where(tbins <= 0 | (N <= 0))] = 1
            # tbins[np.where(tbins >= n | (N >= 1))] = n - 1
            tbins[np.where(np.bitwise_or(tbins <= 0, (N <= 0)))] = 1
            tbins[np.where(np.bitwise_or(tbins >= n, (N >= 1)))] = n - 1

            # s = np.divide((N - cumarc[tbins]), chordlen[tbins - 1])
            # pt = pxy[tbins, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)
            s = np.divide((N - cumarc[tbins - 1]), chordlen[tbins - 1])
            pt = pxy[tbins - 1, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)

            pt[0] = np.array([px[0], py[0]])
            pt[-1] = np.array([px[-1], py[-1]])
            return pt
        else:
            return np.array([(px[0], py[0]),(px[-1], py[-1])])

    
    def __build_scc_tree(self, interp_tree) -> list[list, list]:
        tree_dist = [2, 1, math.dist(interp_tree[2], interp_tree[3])]
        tree_scc = [2, 1, self.__get_slope_change(None, interp_tree[2], interp_tree[3])]
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
                    alpha = self.__get_slope_change(last, current, next)
                dist = math.dist(current, next)
                tree_scc.append(alpha)
                tree_dist.append(dist)

                last = current
                current = next

        tree_scc.append(-tree_scc[2])
        tree_dist.append(tree_dist[2])
    
        return [tree_scc, tree_dist]


    def __get_slope_change(self, p0, p1, p2):
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


    def __get_size(self):
        if self.tree:
            n = 0        
            for k in range(1, len(self.tree)):
                current = self.tree[k]
                if current <= 1:                                
                    n += 1                    
            return n