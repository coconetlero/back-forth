import codecs
import cv2
import yaml
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import image

import numpy as np
import os

import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

from Tortuosity_Measures import TortuosityMeasures as tort



# matplotlib.use('MacOSX')
from matplotlib import image

matplotlib.use('Qt5Agg')


def tree_traversal(image, root, max_distance):
    """
    Traverse the tree into the given image
    :param image: image where the tree is represented
    :param root: Starting position into the image (y,x)
    :param max_distance: maximum distance to obtain the points
    """
    out_positions = [root]
    scc = [0]
    image = np.array(o_image)
    sp = root  # Starting point
    cp = root  # moving Position
    p_idx = 0
    distance = 0
    p_vec = ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))
    while True:
        for i in range(8):
            k = (p_idx + i) % 8
            p = p_vec[k]
            tp = np.array((cp[0] + p[0], cp[1] + p[1]))

            if tp[0] < len(image[0]) and tp[1] < len(image[1]):
                if image[tp[0]][tp[1]] != 0:
                    if k % 2 != 0:
                        kk = (k + 1) % 8
                        if image[cp[0] + p_vec[kk][0]][cp[1] + p_vec[kk][1]] != 0:
                            tp = np.array((cp[0] + p_vec[kk][0], cp[1] + p_vec[kk][1]))
                    distance = np.linalg.norm(tp - sp)
                    cp = tp
                    p_idx = k - 2
                    break

        if distance >= max_distance:
            if len(out_positions) == 1:
                alpha = get_slope_change(out_positions[0], sp, cp)
            else:
                alpha = get_slope_change(out_positions[-2], sp, cp)

            scc.append(alpha)
            out_positions.append(cp)
            sp = cp

        if (cp == root).all():
            break

    return scc


def build_tree(_image, root):
    image = np.array(_image)
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

                    is_special = is_special_point(cp, image)
                    if is_special[0]:
                        if is_special[1] > 2:
                            if is_line_junction(cp, image):
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


def build_interpolated_tree(tree_path):
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
                    D = interp_curve(round(len(bx) * 0.25), by, bx)
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


def build_scc_tree(interp_tree):
    tree_dist = [2, 1, math.dist(interp_tree[2], interp_tree[3])]
    tree_scc = [2, 1, get_slope_change(None, interp_tree[2], interp_tree[3])]
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
                alpha = get_slope_change(last, current, next)
            dist = math.dist(current, next)
            tree_scc.append(alpha)
            tree_dist.append(dist)

            last = current
            current = next

    tree_scc.append(-tree_scc[2])
    tree_dist.append(tree_dist[2])
    
    return [tree_scc, tree_dist]


def build_parentheses_tree(interp_tree):
    """

    :param interp_tree:
    :return:
    """
    p1 = 2
    p2 = 0
    current = interp_tree[2]
    tree = "2({X},{Y})".format(X=current[0], Y=current[1])

    for k in range(3, len(interp_tree)-1):
        last = current
        current = interp_tree[k]

        if type(current) is not tuple:
            if current != 1:
                if not p1:
                    p1 = current
                else:
                    p2 = current
            if p1 and p2:
                if p1 < p2:
                    node = "({n}({X},{Y})".format(n=current, X=last[0], Y=last[1])
                else:
                    node = ")"

                p1 = p2
                p2 = 0
                tree += node
        else:
            continue

    return tree


def build_full_parentheses_tree_bkup(interp_tree):
    """

    :param interp_tree:
    :return:
    """
    p1 = 2
    p2 = 0
    max_p = p1

    current = interp_tree[2]
    next = interp_tree[3]
    tree = "2({X},{Y}),".format(X=current[0], Y=current[1])
    branch = ""
    special = {}
    i = 3
    il = 0
    for k in range(3, len(interp_tree)-1):
        last = current
        current = next
        next = interp_tree[k+1]

        if type(current) is not tuple:
            if current != 1:
                if not special.get(current):
                    special[current] = i

                if not p1:
                    p1 = special[current]
                else:
                    p2 = special[current]

                if p2 > max_p:
                    max_p = p2

            if p1 and p2:
                if p1 < p2:
                    node = "{n}({X:.0f},{Y:.0f})".format(n=special[current], X=last[0], Y=last[1])

                    if next == 1:
                        branch += node + ")\n"
                    else:
                        branch += node + "\n"
                    i += 1
                    il = i
                else:
                    i = il
                    branch = ""
                    if (max_p - p1) >= 2:
                        branch = ")"

                tree += branch
                branch = "("

                p1 = p2
                p2 = 0

        else:
            if type(next) is tuple:
                node = "{n}({X:.1f},{Y:.1f}),".format(n=i, X=current[0], Y=current[1])
                branch += node
                i += 1

    return tree


def build_full_parentheses_tree(interp_tree):
    """

        :param interp_tree:
        :return:
        """
    p1 = 2
    p2 = 0
    max_p = p1 

    last = interp_tree[2]
    current = interp_tree[2]
    tree = ""
    branch = ""
    i = 3
    il = 0
    for k in range(3, len(interp_tree) - 1):
        next = interp_tree[k]
        if type(next) is not tuple:
            if next == 1:
                tree = tree[:-2] + ")\n"
            else:
                if not p1:
                    p1 = next
                else:
                    p2 = next

                if p2 > max_p:
                    max_p = p2

            if p1 and p2:
                if p1 < p2:
                    branch = "({p1}-{p2})".format(p1=p1, p2=p2) + branch[:-1] + "\n"
                    il = i
                else:
                    i = il
                    branch = ""
                    if (max_p - p1) >= 2:
                        tree = tree[:-1] + ")\n"

                tree += branch
                branch = "("

                p1 = p2
                p2 = 0
        else:
            node = "({X:.2f},{Y:.2f}), ".format(n=i, X=current[0], Y=current[1])
            branch += node
            i += 1

            last = current
            current = next

    return tree


def build_scc_parentheses_tree(interp_tree):
    """

    :param interp_tree:
    :return:
    """
    p1 = 2
    p2 = 0
    max_p = p1

    last = interp_tree[2]
    current = interp_tree[2]
    tree = ""
    branch = ""
    i = 3
    il = 0
    for k in range(3, len(interp_tree) - 1):
        next = interp_tree[k]
        if type(next) is not tuple:
            if next == 1:
                tree = tree[:-2] + ")\n"
            else:
                if not p1:
                    p1 = next
                else:
                    p2 = next

                if p2 > max_p:
                    max_p = p2

            if p1 and p2:
                if p1 < p2:
                    branch = "({p1}-{p2})".format(p1=p1, p2=p2) + branch[:-1] + "\n"
                    il = i
                else:
                    i = il
                    branch = ""
                    if (max_p - p1) >= 2:
                        tree = tree[:-1] + ")\n"

                tree += branch
                branch = "("

                p1 = p2
                p2 = 0
        else:
            alpha = get_slope_change(last, current, next)
            node = "{s:.4f}, ".format(n=i, s=alpha)
            branch += node
            i += 1

            last = current
            current = next

    return tree


def find_branches(tree_path):
    p1 = 0
    p2 = 0
    branch = []
    branches = {}
    for data in tree_path:
        if type(data) is not tuple:
            if data != 1:
                if not p1:
                    p1 = data
                else:
                    p2 = data
            if p1 and p2:
                if p1 < p2:
                    branches[(p1, p2)] = branch

                branch = [branch[-1]]
                p1 = p2
                p2 = 0
        else:
            branch.append(data)

    return branches


def rebuild_final_tree(branches):
    for br in branches.values():
        bx = np.array([point[1] for point in br])
        by = np.array([point[0] for point in br])

        D = interp_curve(round(len(bx) * 0.25), by, bx)



        fig, ax = plt.subplots()
        plt.grid(True)
        ax.axis('equal')

        ax.plot(bx, by, 'r.-', linewidth=2.0)
        ax.plot(D[:, 1], D[:, 0], 'b.-', linewidth=2.0)

        plt.show()

        print(math.dist((D[0, 0], D[0, 1]), (D[1, 0], D[1, 1])))


def is_special_point(p, image):
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


def is_line_junction(p, image):
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


def get_slope_change(p0, p1, p2):
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


def interp_curve(n, px, py):
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


def show_plot(doc) -> None:
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(doc.modelspace(), finalize=True)

    # fig.show()
    plt.show()


def display_tree(scc_tree, dist_tree):
    if not dist_tree:
        dist_tree = [1] * len(scc_tree)

    slope = 0
    segment_size = 1
    xx1 = -8
    yy1 = 0
    points = []

    k = 2
    while k < len(scc_tree):
        val = scc_tree[k]
        segment_size = dist_tree[k]

        if val < 2:
            p = (xx1, yy1)
            points.append(p)

            s = np.double(val)
            slope = slope + s
            xx1 = xx1 + segment_size * np.cos(slope * np.pi)
            yy1 = yy1 + segment_size * np.sin(slope * np.pi)
        k += 1


    doc = ezdxf.new('R12')
    msp = doc.modelspace()

    polyline = msp.add_polyline3d([])
    polyline.append_vertices(points)

    show_plot(doc)




def display_tree_2(scc_filename, dist_filename):
    scc = []
    dist = []

    first = True
    with open(scc_filename) as scc_file:
        for line in scc_file:
            line = line.rstrip("\n")
            data = line.split(" ")

            if first:
                # remove first element of the point list
                data.pop(0)
                first = False

            for val in data:
                if not val == '':
                    s = np.double(val)
                    scc.append(s)

    first = True
    with open(dist_filename) as dist_file:
        for line in dist_file:
            line = line.rstrip("\n")
            data = line.split(" ")

            if first:
                # remove first element of the point list
                data.pop(0)
                first = False

            for val in data:
                if not val == '':
                    s = np.double(val)
                    dist.append(s)

    slope = 0
    segment_size = 1
    xx1 = -8
    yy1 = 0
    points = []

    X = []
    Y = []

    k = 2
    while k < len(scc):
        val = scc[k]
        segment_size = dist[k]

        if val < 2:
            X.append(xx1)
            Y.append(yy1)

            p = (xx1, yy1)
            points.append(p)

            s = np.double(val)
            slope = slope + s
            xx1 = xx1 + segment_size * np.cos(slope * np.pi)
            yy1 = yy1 + segment_size * np.sin(slope * np.pi)
        k += 1

    # plt.plot(X, Y, 'r.-')
    # plt.show()

    doc = ezdxf.new('R12')
    msp = doc.modelspace()

    polyline = msp.add_polyline3d([])
    polyline.append_vertices(points)

    show_plot(doc)


def plot_tree(scc_tree, dist_tree):

    slope = 0
    segment_size = 1
    xx1 = -8
    yy1 = 0
    points = []
    X = []
    Y = []
    k = 2
    while k < len(scc_tree):
        val = scc_tree[k]
        segment_size = dist_tree[k]

        if val < 2:
            X.append(xx1)
            Y.append(yy1)

            s = np.double(val)
            slope = slope + s
            xx1 = xx1 + segment_size * np.cos(slope * np.pi)
            yy1 = yy1 + segment_size * np.sin(slope * np.pi)
        k += 1

    plt.plot(X, Y, 'r.-')
    plt.axis('equal')
    plt.show()

    return [X, Y]


def write_scc_file(scc_filename, dist_filename, ssc_tree, dist_tree):
    scc_file = codecs.open(scc_filename, "w+", "utf-8")
    scc_file.write('{}'.format(len(ssc_tree) + 1))

    dist_file = codecs.open(dist_filename, "w+", "utf-8")
    dist_file.write('{}'.format(len(dist_tree) + 1))

    for cur, nxt in zip(ssc_tree, ssc_tree[1:]):
        if cur >= 1.0 or cur == -1.0:
            if nxt >= 1.0 or nxt == -1.0:
                scc_file.write('\n')
                scc_file.write('{:.1f} '.format(cur))
            else:
                scc_file.write('{:.1f}\n'.format(cur))
        else:
            scc_file.write('{:.5f} '.format(cur))

    for cur, nxt in zip(dist_tree, dist_tree[1:]):
        if cur >= 1.0 or cur == -1.0:
            if nxt >= 1.0 or nxt == -1.0:
                dist_file.write('\n')
                dist_file.write('{:.5f} '.format(cur))
            else:
                dist_file.write('{:.5f}\n'.format(cur))
        else:
            dist_file.write('{:.5f} '.format(cur))

    scc_file.write('1.00000\n')
    scc_file.close()

    dist_file.write('1.00000\n')
    dist_file.close()


def create_scc_closed_curve(dx, dy):
    scc_curve = []
    last = [dy[0],dx[0]]
    current = [dy[1],dx[1]]
    for k in range(2, len(dx)):        
        next = [dy[k],dx[k]]
        slope_d = get_slope_change(last, current, next)
        scc_curve.append(slope_d)

        last = current
        current = next
    return scc_curve 


if __name__ == '__main__':


    with open('./config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)

    # Elena images
    # sp = (212,362)
    sp = (config_data["start_position"]["x"],config_data["start_position"]["y"])
    o_image = cv2.imread(config_data["base_folder"] + config_data["binary_image"], cv2.IMREAD_GRAYSCALE)
    o_file = config_data["base_folder"] + config_data["output_file"]
    d_file = config_data["base_folder"] + config_data["distances_file"]




    # tree_traversal(o_image, sp, 1)
    # [scc, dist] = build_scc_tree(treepath)

    # display_tree_2(o_file, d_file)

    # display_tree_2('/Users/zianfanti/IIMAS/Three_Representation/data/trees/DB_01/healthy_bw/07_h_bin_v_01.scc',
    #                '/Users/zianfanti/IIMAS/Three_Representation/data/trees/DB_01/healthy_bw/07_h_bin_v_01_d.scc')

    treepath = build_tree(o_image, sp)
    interp_tree = build_interpolated_tree(treepath)
    [scc, dist] = build_scc_tree(interp_tree)

    # tort.SCC(scc)

    display_tree(scc, dist)
    [X, Y] = plot_tree(scc, dist)
    write_scc_file(o_file, d_file, scc, dist)

    # tree = build_full_parentheses_tree(interp_tree)
    tree = build_scc_parentheses_tree(interp_tree)
    print(tree)

    # branches = find_branches(treepath)

    # rebuild_final_tree(branches)

    # alpha = tree_traversal(o_image, sp, max_distance)
